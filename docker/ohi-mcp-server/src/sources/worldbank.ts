/**
 * World Bank Source
 * =================
 * 
 * Economic indicators.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

type WorldBankResponse = [
  { page: number; pages: number; total: number },
  Array<{
    indicator?: { id?: string; value?: string };
    country?: { id?: string; value?: string };
    date?: string;
    value?: number | null;
  }> | null
];

export class WorldBankSource extends BaseSource {
  name = "worldbank";
  description = "World Bank economic indicators";

  constructor() {
    super("https://api.worldbank.org/v2");
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/country/US/indicator/NY.GDP.MKTP.CD`, {
        params: { format: "json", per_page: 1 },
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, _limit = 5): Promise<SearchResult[]> {
    // Try to parse indicator from query
    const indicator = this.extractIndicator(query) || "NY.GDP.MKTP.CD";
    return this.getIndicator(indicator, "all");
  }

  async getIndicator(
    indicator: string,
    country = "all",
    year?: string
  ): Promise<SearchResult[]> {
    try {
      const params: Record<string, string | number> = {
        format: "json",
        per_page: 50,
      };

      if (year) {
        if (year.includes(":")) {
          params.date = year; // Range: 2015:2020
        } else {
          params.date = year;
        }
      }

      const response = await httpClient.get<WorldBankResponse>(
        `${this.baseUrl}/country/${country}/indicator/${indicator}`,
        { params }
      );

      const [, data] = response.data;
      if (!data) return [];

      // Group by country if 'all'
      const byCountry = new Map<string, Array<{ date: string; value: number | null }>>();
      
      for (const entry of data) {
        const countryName = entry.country?.value || "Unknown";
        if (!byCountry.has(countryName)) {
          byCountry.set(countryName, []);
        }
        byCountry.get(countryName)!.push({
          date: entry.date || "",
          value: entry.value ?? null,
        });
      }

      const results: SearchResult[] = [];
      
      for (const [countryName, values] of byCountry) {
        const validValues = values.filter((v) => v.value !== null);
        if (validValues.length === 0) continue;

        const latestValue = validValues[0];
        const indicatorName = data[0]?.indicator?.value || indicator;

        results.push({
          source: this.name,
          title: `${indicatorName} - ${countryName}`,
          content: validValues
            .slice(0, 10)
            .map((v) => `${v.date}: ${this.formatValue(v.value)}`)
            .join("\n"),
          url: `https://data.worldbank.org/indicator/${indicator}`,
          metadata: {
            indicator,
            country: countryName,
            latest_year: latestValue.date,
            latest_value: latestValue.value,
          },
        });
      }

      return results.slice(0, 10);
    } catch {
      return [];
    }
  }

  private formatValue(value: number | null): string {
    if (value === null) return "N/A";
    if (Math.abs(value) >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
    if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    return value.toLocaleString();
  }

  private extractIndicator(query: string): string | null {
    // Common indicator patterns
    const patterns: Record<string, string> = {
      gdp: "NY.GDP.MKTP.CD",
      "gdp per capita": "NY.GDP.PCAP.CD",
      population: "SP.POP.TOTL",
      "life expectancy": "SP.DYN.LE00.IN",
      inflation: "FP.CPI.TOTL.ZG",
      unemployment: "SL.UEM.TOTL.ZS",
      co2: "EN.ATM.CO2E.PC",
    };

    const lowerQuery = query.toLowerCase();
    for (const [key, value] of Object.entries(patterns)) {
      if (lowerQuery.includes(key)) return value;
    }

    // Check if query is already an indicator code
    if (/^[A-Z]{2}\.[A-Z]{3}/.test(query)) return query;

    return null;
  }
}
