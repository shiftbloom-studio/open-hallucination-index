from __future__ import annotations

import faulthandler
import logging
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict
from typing import Any, Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Qt, Signal, QTimer
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ingestion.models import IngestionConfig, PipelineStats
from ingestion.pipeline import IngestionPipeline

# Enable faulthandler for crash diagnostics (segfaults, etc.)
faulthandler.enable()

# Crash log file for post-mortem debugging
CRASH_LOG_FILE = ".ingestion_crash.log"


def _write_crash_log(message: str) -> None:
    """Write crash information to a file for post-mortem debugging."""
    try:
        with open(CRASH_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Crash at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
            f.write(message)
            f.write("\n")
    except Exception:
        pass  # Don't crash while logging a crash


class _QtLogHandler(logging.Handler):
    def __init__(self, emit: Callable[[str], None]) -> None:
        super().__init__()
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._emit(msg)
        except Exception:
            pass  # Don't crash if GUI is shutting down


class IngestionWorker(QThread):
    stats_updated = Signal(dict)
    log_message = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)
    connection_warning = Signal(str)  # Signal for connection issues
    heartbeat = Signal()  # New: heartbeat signal to indicate worker is alive

    # Watchdog timeout in seconds - if no heartbeat for this long, worker is considered hung
    WATCHDOG_TIMEOUT = 120.0  # 2 minutes without activity = likely hung

    def __init__(
        self,
        config: IngestionConfig,
        no_resume: bool,
        clear_checkpoint: bool,
    ) -> None:
        super().__init__()
        self._config = config
        self._no_resume = no_resume
        self._clear_checkpoint = clear_checkpoint
        self._pipeline: IngestionPipeline | None = None
        self._last_heartbeat = time.time()
        self._force_stop = False

    def stop(self) -> None:
        """Request graceful shutdown."""
        if self._pipeline:
            self._pipeline.shutdown()

    def force_stop(self) -> None:
        """Force immediate termination (for hung worker recovery)."""
        self._force_stop = True
        self.stop()

    def _emit_heartbeat(self) -> None:
        """Emit heartbeat to indicate worker is alive."""
        self._last_heartbeat = time.time()
        try:
            self.heartbeat.emit()
        except Exception:
            pass  # GUI might be shutting down

    def run(self) -> None:
        handler = _QtLogHandler(self.log_message.emit)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        try:
            self._run_internal()
        except Exception as exc:  # pragma: no cover - GUI worker safety net
            error_msg = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            _write_crash_log(f"Worker crashed:\n{error_msg}\n\nTraceback:\n{tb}")
            self.failed.emit(error_msg)
        finally:
            try:
                root_logger.removeHandler(handler)
            except Exception:
                pass

    def _run_internal(self) -> None:
        """Internal run method with full exception handling."""
        self._emit_heartbeat()
        
        try:
            self._pipeline = IngestionPipeline(self._config)
        except Exception as exc:
            error_msg = f"Failed to initialize pipeline: {exc}"
            _write_crash_log(error_msg)
            self.failed.emit(error_msg)
            return
        
        self._emit_heartbeat()

        # Test connections before starting
        if not self._check_connections():
            self.failed.emit("Connection check failed. Please verify Qdrant and Neo4j are accessible.")
            return

        self._emit_heartbeat()

        try:
            self._pipeline.load_checkpoint(
                no_resume=self._no_resume,
                clear=self._clear_checkpoint,
            )
        except Exception as exc:
            error_msg = f"Failed to load checkpoint: {exc}"
            _write_crash_log(error_msg)
            self.failed.emit(error_msg)
            return

        if self._clear_checkpoint:
            stats = PipelineStats()
            self.stats_updated.emit(self._snapshot(stats))
            self.finished.emit(self._snapshot(stats))
            return

        def on_progress(stats: PipelineStats) -> None:
            self._emit_heartbeat()
            self.stats_updated.emit(self._snapshot(stats))

        try:
            stats = self._pipeline.run(
                limit=self._config.limit,
                progress_callback=on_progress,
            )
            self._emit_heartbeat()
            self.stats_updated.emit(self._snapshot(stats))
            self.finished.emit(self._snapshot(stats))
        except Exception as exc:
            error_msg = f"Pipeline run failed: {type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            _write_crash_log(f"{error_msg}\n\nTraceback:\n{tb}")
            self.failed.emit(error_msg)

    def _check_connections(self) -> bool:
        """
        Check if Qdrant and Neo4j connections are working.
        
        Returns:
            True if both connections are healthy, False otherwise
        """
        try:
            # Check Qdrant
            from qdrant_client import QdrantClient
            qdrant = QdrantClient(
                host=self._config.qdrant_host,
                port=self._config.qdrant_port,
                timeout=10,
                https=self._config.qdrant_https,
                verify=self._config.qdrant_tls_ca_cert,
            )

            qdrant.get_collections()
            self.log_message.emit("✅ Qdrant connection OK")
        except Exception as e:
            self.log_message.emit(f"❌ Qdrant connection failed: {e}")
            return False

        try:
            # Check Neo4j
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self._config.neo4j_uri,
                auth=(self._config.neo4j_user, self._config.neo4j_password),
                connection_acquisition_timeout=10,
            )
            with driver.session(database=self._config.neo4j_database) as session:
                session.run("RETURN 1")
            driver.close()
            self.log_message.emit("✅ Neo4j connection OK")
        except Exception as e:
            self.log_message.emit(f"❌ Neo4j connection failed: {e}")
            return False

        return True

    @staticmethod
    def _snapshot(stats: PipelineStats) -> dict[str, Any]:
        data = asdict(stats)
        data["rate"] = stats.rate()
        data["timestamp"] = time.time()
        return data


class IngestionWindow(QMainWindow):
    # Watchdog timeout - if no heartbeat for this long, show warning
    WATCHDOG_WARNING_TIMEOUT = 60.0  # 1 minute without heartbeat = warning
    WATCHDOG_CRITICAL_TIMEOUT = 180.0  # 3 minutes = offer force stop

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OHI Ingestion Studio")
        self.resize(1280, 820)

        self._worker: IngestionWorker | None = None
        self._last_plot_update: float = 0.0
        self._last_heartbeat: float = 0.0
        self._watchdog_warned: bool = False

        self._time_series = deque(maxlen=600)
        self._rate_series = deque(maxlen=600)
        self._articles_series = deque(maxlen=600)
        self._chunks_series = deque(maxlen=600)
        self._errors_series = deque(maxlen=600)
        self._download_q_series = deque(maxlen=600)
        self._preprocess_q_series = deque(maxlen=600)
        self._upload_q_series = deque(maxlen=600)

        # Watchdog timer to detect hung workers
        self._watchdog_timer = QTimer(self)
        self._watchdog_timer.timeout.connect(self._check_watchdog)
        self._watchdog_timer.setInterval(5000)  # Check every 5 seconds

        self._build_ui()
        self._apply_dark_theme()
        
        # Check for previous crash on startup
        self._check_previous_crash()

    def _apply_dark_theme(self) -> None:
        QApplication.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(20, 22, 28))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(30, 32, 40))
        palette.setColor(QPalette.AlternateBase, QColor(44, 48, 60))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(45, 48, 58))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor(88, 166, 255))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(palette)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_visual_panel())
        splitter.setSizes([420, 860])

        layout.addWidget(splitter)
        self.setCentralWidget(root)

    def _build_config_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        config_group = QGroupBox("Ingestion Configuration")
        form = QFormLayout(config_group)

        self.qdrant_host = QLineEdit("localhost")
        self.qdrant_port = QSpinBox()
        self.qdrant_port.setRange(1, 65535)
        self.qdrant_port.setValue(6333)
        self.qdrant_grpc_port = QSpinBox()
        self.qdrant_grpc_port.setRange(1, 65535)
        self.qdrant_grpc_port.setValue(6334)
        self.qdrant_https = QCheckBox("Use TLS")
        self.qdrant_tls_ca_cert = QLineEdit("")
        self.qdrant_collection = QLineEdit("wikipedia_hybrid")


        self.neo4j_uri = QLineEdit("bolt://localhost:7687")
        self.neo4j_user = QLineEdit("neo4j")
        self.neo4j_pass = QLineEdit("password123")
        self.neo4j_pass.setEchoMode(QLineEdit.Password)
        self.neo4j_db = QLineEdit("neo4j")

        self.limit = QSpinBox()
        self.limit.setRange(0, 100000000)
        self.limit.setValue(0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(16, 4096)
        self.batch_size.setValue(384)
        self.embedding_batch_size = QSpinBox()
        self.embedding_batch_size.setRange(32, 8192)
        self.embedding_batch_size.setValue(768)
        self.embedding_device = QComboBox()
        self.embedding_device.addItems(["auto", "cuda", "cpu"])

        self.download_dir = QLineEdit(".wiki_dumps")
        self.keep_downloads = QCheckBox("Keep downloads")
        self.no_resume = QCheckBox("No resume (reset stats)")
        self.clear_checkpoint = QCheckBox("Clear checkpoint")

        form.addRow("Qdrant host", self.qdrant_host)
        form.addRow("Qdrant port", self.qdrant_port)
        form.addRow("Qdrant gRPC", self.qdrant_grpc_port)
        form.addRow("Qdrant TLS", self.qdrant_https)
        form.addRow("Qdrant TLS CA", self.qdrant_tls_ca_cert)
        form.addRow("Qdrant collection", self.qdrant_collection)

        form.addRow("Neo4j URI", self.neo4j_uri)
        form.addRow("Neo4j user", self.neo4j_user)
        form.addRow("Neo4j password", self.neo4j_pass)
        form.addRow("Neo4j database", self.neo4j_db)
        form.addRow("Limit (0 = no limit)", self.limit)
        form.addRow("Batch size", self.batch_size)
        form.addRow("Embedding batch", self.embedding_batch_size)
        form.addRow("Embedding device", self.embedding_device)
        form.addRow("Download dir", self.download_dir)
        form.addRow("", self.keep_downloads)
        form.addRow("", self.no_resume)
        form.addRow("", self.clear_checkpoint)

        layout.addWidget(config_group)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Ingestion")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.recover_btn = QPushButton("Recover from DB")
        self.recover_btn.setToolTip(
            "Scan Qdrant and Neo4j to recover checkpoint from existing data.\n"
            "Use this if the checkpoint file was lost but data exists in databases."
        )
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.recover_btn)
        layout.addLayout(btn_row)

        self.status_label = QLabel("Idle")
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.start_btn.clicked.connect(self._start_ingestion)
        self.stop_btn.clicked.connect(self._stop_ingestion)
        self.recover_btn.clicked.connect(self._recover_checkpoint)

        return panel

    def _build_visual_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_charts_tab(), "Live Charts")
        self.tabs.addTab(self._build_log_tab(), "Logs")

        layout.addWidget(self.tabs)
        return panel

    def _build_charts_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax_rate = self.figure.add_subplot(2, 2, 1)
        self.ax_articles = self.figure.add_subplot(2, 2, 2)
        self.ax_chunks = self.figure.add_subplot(2, 2, 3)
        self.ax_queues = self.figure.add_subplot(2, 2, 4)

        self._line_rate, = self.ax_rate.plot([], [], color="#58a6ff", label="articles/sec")
        self._line_articles, = self.ax_articles.plot([], [], color="#7ee787", label="articles")
        self._line_chunks, = self.ax_chunks.plot([], [], color="#f778ba", label="chunks")
        self._line_dl, = self.ax_queues.plot([], [], color="#ffb347", label="download q")
        self._line_pre, = self.ax_queues.plot([], [], color="#8bd5ff", label="preprocess q")
        self._line_up, = self.ax_queues.plot([], [], color="#ffa657", label="upload q")

        for ax, title in [
            (self.ax_rate, "Throughput"),
            (self.ax_articles, "Articles Processed"),
            (self.ax_chunks, "Chunks Created"),
            (self.ax_queues, "Queue Depths"),
        ]:
            ax.set_title(title)
            ax.grid(True, alpha=0.2)

        self.ax_queues.legend(loc="upper left")

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        return container

    def _build_log_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        return container

    def _start_ingestion(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        config = IngestionConfig(
            qdrant_host=self.qdrant_host.text().strip(),
            qdrant_port=self.qdrant_port.value(),
            qdrant_grpc_port=self.qdrant_grpc_port.value(),
            qdrant_https=self.qdrant_https.isChecked(),
            qdrant_tls_ca_cert=self.qdrant_tls_ca_cert.text().strip() or None,
            qdrant_collection=self.qdrant_collection.text().strip(),

            neo4j_uri=self.neo4j_uri.text().strip(),
            neo4j_user=self.neo4j_user.text().strip(),
            neo4j_password=self.neo4j_pass.text(),
            neo4j_database=self.neo4j_db.text().strip(),
            limit=self.limit.value() or None,
            batch_size=self.batch_size.value(),
            embedding_batch_size=self.embedding_batch_size.value(),
            embedding_device=self.embedding_device.currentText(),
            download_dir=self.download_dir.text().strip(),
            keep_downloads=self.keep_downloads.isChecked(),
        )

        self._worker = IngestionWorker(
            config=config,
            no_resume=self.no_resume.isChecked(),
            clear_checkpoint=self.clear_checkpoint.isChecked(),
        )
        self._worker.stats_updated.connect(self._on_stats_update)
        self._worker.log_message.connect(self._append_log)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.heartbeat.connect(self._on_heartbeat)

        # Reset watchdog state
        self._last_heartbeat = time.time()
        self._watchdog_warned = False
        self._watchdog_timer.start()

        self.status_label.setText("Running...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._worker.start()

    def _stop_ingestion(self) -> None:
        if self._worker:
            self._worker.stop()
            self.status_label.setText("Stopping...")

    def _force_stop_ingestion(self) -> None:
        """Force stop a hung worker."""
        if self._worker:
            self._append_log("⚠️ Force stopping worker...")
            self._worker.force_stop()
            self._watchdog_timer.stop()
            # Give it a moment to clean up
            QTimer.singleShot(2000, self._check_force_stop_complete)

    def _check_force_stop_complete(self) -> None:
        """Check if force stop completed and reset UI if needed."""
        if self._worker and self._worker.isRunning():
            self._append_log("⚠️ Worker did not respond to force stop. You may need to restart the application.")
        self._reset_ui_after_stop()

    def _reset_ui_after_stop(self) -> None:
        """Reset UI state after worker stops."""
        self._watchdog_timer.stop()
        self.status_label.setText("Stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_heartbeat(self) -> None:
        """Handle heartbeat from worker."""
        self._last_heartbeat = time.time()
        self._watchdog_warned = False

    def _check_watchdog(self) -> None:
        """Check if worker is still responsive."""
        if not self._worker or not self._worker.isRunning():
            self._watchdog_timer.stop()
            return

        elapsed = time.time() - self._last_heartbeat

        if elapsed > self.WATCHDOG_CRITICAL_TIMEOUT:
            self._watchdog_timer.stop()
            response = QMessageBox.warning(
                self,
                "Worker Not Responding",
                f"The ingestion worker has not responded for {int(elapsed)} seconds.\n\n"
                "This may indicate a hang or crash.\n\n"
                "Do you want to force stop the worker?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if response == QMessageBox.Yes:
                self._force_stop_ingestion()
            else:
                # Resume watching but reset heartbeat to avoid immediate re-prompt
                self._last_heartbeat = time.time()
                self._watchdog_timer.start()
        elif elapsed > self.WATCHDOG_WARNING_TIMEOUT and not self._watchdog_warned:
            self._watchdog_warned = True
            self._append_log(
                f"⚠️ No activity for {int(elapsed)} seconds - worker may be processing a large batch or experiencing issues"
            )

    def _check_previous_crash(self) -> None:
        """Check for crash log from previous session."""
        if os.path.exists(CRASH_LOG_FILE):
            try:
                with open(CRASH_LOG_FILE, "r", encoding="utf-8") as f:
                    crash_info = f.read()
                if crash_info.strip():
                    # Show a brief notification about the crash log
                    self._append_log(
                        "⚠️ Previous crash detected. See .ingestion_crash.log for details."
                    )
            except Exception:
                pass

    def _on_failed(self, message: str) -> None:
        self._watchdog_timer.stop()
        self.status_label.setText("Failed")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append_log(f"❌ {message}")

    def _on_finished(self, stats: dict) -> None:
        self._watchdog_timer.stop()
        self.status_label.setText("Completed")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append_log("✅ Ingestion finished.")
        self._on_stats_update(stats)

    def _append_log(self, message: str) -> None:
        self.log_output.append(message)

    def _on_stats_update(self, stats: dict) -> None:
        now = stats.get("timestamp", time.time())
        if now - self._last_plot_update < 0.8:
            return
        self._last_plot_update = now

        self._time_series.append(now)
        self._rate_series.append(stats.get("rate", 0.0))
        self._articles_series.append(stats.get("articles_processed", 0))
        self._chunks_series.append(stats.get("chunks_created", 0))
        self._errors_series.append(stats.get("errors", 0))
        self._download_q_series.append(stats.get("download_queue_depth", 0))
        self._preprocess_q_series.append(stats.get("preprocess_queue_depth", 0))
        self._upload_q_series.append(stats.get("upload_queue_depth", 0))

        self._refresh_plots()

    def _refresh_plots(self) -> None:
        if not self._time_series:
            return
        xs = [t - self._time_series[0] for t in self._time_series]

        self._line_rate.set_data(xs, list(self._rate_series))
        self._line_articles.set_data(xs, list(self._articles_series))
        self._line_chunks.set_data(xs, list(self._chunks_series))
        self._line_dl.set_data(xs, list(self._download_q_series))
        self._line_pre.set_data(xs, list(self._preprocess_q_series))
        self._line_up.set_data(xs, list(self._upload_q_series))

        for ax in [self.ax_rate, self.ax_articles, self.ax_chunks, self.ax_queues]:
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw_idle()

    def _recover_checkpoint(self) -> None:
        """Recover checkpoint by scanning existing data in Qdrant and Neo4j."""
        response = QMessageBox.question(
            self,
            "Recover Checkpoint",
            "This will scan Qdrant and Neo4j to find already-ingested articles\n"
            "and update the checkpoint file to prevent duplicates.\n\n"
            "This can take several minutes for large datasets.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if response != QMessageBox.Yes:
            return

        self.status_label.setText("Recovering checkpoint...")
        self.start_btn.setEnabled(False)
        self.recover_btn.setEnabled(False)
        self._append_log("🔄 Starting checkpoint recovery from databases...")

        # Run in a thread to avoid blocking GUI
        from threading import Thread
        
        def do_recovery() -> None:
            try:
                # Create config from GUI
                config = IngestionConfig(
                    qdrant_host=self.qdrant_host.text(),
                    qdrant_port=self.qdrant_port.value(),
                    qdrant_grpc_port=self.qdrant_grpc_port.value(),
                    qdrant_collection=self.qdrant_collection.text(),
                    qdrant_https=self.qdrant_https.isChecked(),
                    qdrant_tls_ca_cert=self.qdrant_tls_ca_cert.text() or None,
                    neo4j_uri=self.neo4j_uri.text(),
                    neo4j_user=self.neo4j_user.text(),
                    neo4j_password=self.neo4j_pass.text(),
                    neo4j_database=self.neo4j_db.text(),
                    download_dir=self.download_dir.text(),
                    limit=self.limit.value() or None,
                    batch_size=self.batch_size.value(),
                    embedding_batch_size=self.embedding_batch_size.value(),
                    embedding_device=self.embedding_device.currentText(),
                    keep_downloads=self.keep_downloads.isChecked(),
                )
                
                pipeline = IngestionPipeline(config)
                recovered = pipeline.recover_checkpoint_from_databases()
                
                # Clean up
                pipeline.qdrant.close()
                pipeline.neo4j.close()
                
                # Update GUI on main thread
                self._recovery_complete(recovered, None)
                
            except Exception as e:
                self._recovery_complete(0, str(e))
        
        Thread(target=do_recovery, daemon=True).start()

    def _recovery_complete(self, recovered: int, error: str | None) -> None:
        """Handle recovery completion - called from background thread."""
        def update_gui() -> None:
            self.start_btn.setEnabled(True)
            self.recover_btn.setEnabled(True)
            
            if error:
                self.status_label.setText("Recovery failed")
                self._append_log(f"❌ Recovery failed: {error}")
            else:
                self.status_label.setText("Idle")
                if recovered > 0:
                    self._append_log(f"✅ Recovered {recovered:,} article IDs from databases")
                else:
                    self._append_log("📍 No existing articles found in databases")
        
        # Schedule the update on the main thread
        QTimer.singleShot(0, update_gui)


def main() -> None:
    app = QApplication(sys.argv)
    window = IngestionWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()