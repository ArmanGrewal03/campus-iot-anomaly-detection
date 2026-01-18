<template>
  <div>
    <!-- API Status Alert -->
    <base-alert
      :type="stats && stats.api_online ? 'success' : 'danger'"
      class="mb-3"
    >
      <i
        :class="
          stats && stats.api_online
            ? 'tim-icons icon-check-2'
            : 'tim-icons icon-alert-circle-exc'
        "
      ></i>
      {{
        stats && stats.api_online
          ? $t("dashboard.apiOnline")
          : $t("dashboard.apiOffline")
      }}
    </base-alert>

    <!-- KPI Cards -->
    <div class="row">
      <div class="col-lg-3 col-md-6" :class="{ 'text-right': isRTL }">
        <card>
          <template slot="header">
            <h5 class="card-category">{{ $t("dashboard.totalRecords") }}</h5>
            <h3 class="card-title">
              <i class="tim-icons icon-bullet-list-67 text-primary"></i>
              {{ stats ? stats.total_records : "-" }}
            </h3>
          </template>
        </card>
      </div>
      <div class="col-lg-3 col-md-6" :class="{ 'text-right': isRTL }">
        <card>
          <template slot="header">
            <h5 class="card-category">{{ $t("dashboard.trainingData") }}</h5>
            <h3 class="card-title">
              <i class="tim-icons icon-chart-bar-32 text-info"></i>
              {{ stats ? stats.training_records : "-" }}
            </h3>
            <p class="card-category" v-if="stats && stats.total_records > 0">
              {{ stats.training_percentage }}% {{ $t("dashboard.ofTotal") }}
            </p>
            <div class="progress" style="height: 4px" v-if="stats">
              <div
                class="progress-bar bg-info"
                :style="{
                  width:
                    stats.total_records > 0
                      ? (stats.training_percentage || 0) + '%'
                      : '0%',
                }"
              ></div>
            </div>
          </template>
        </card>
      </div>
      <div class="col-lg-3 col-md-6" :class="{ 'text-right': isRTL }">
        <card>
          <template slot="header">
            <h5 class="card-category">{{ $t("dashboard.testingData") }}</h5>
            <h3 class="card-title">
              <i class="tim-icons icon-send text-success"></i>
              {{ stats ? stats.testing_records : "-" }}
            </h3>
            <p class="card-category" v-if="stats && stats.total_records > 0">
              {{ stats.testing_percentage }}% {{ $t("dashboard.ofTotal") }}
            </p>
            <div class="progress" style="height: 4px" v-if="stats">
              <div
                class="progress-bar bg-success"
                :style="{
                  width:
                    stats.total_records > 0
                      ? (stats.testing_percentage || 0) + '%'
                      : '0%',
                }"
              ></div>
            </div>
          </template>
        </card>
      </div>
      <div class="col-lg-3 col-md-6" :class="{ 'text-right': isRTL }">
        <card>
          <template slot="header">
            <h5 class="card-category">{{ $t("dashboard.apiStatus") }}</h5>
            <h3
              class="card-title"
              :class="
                stats && stats.api_online ? 'text-success' : 'text-danger'
              "
            >
              <i
                :class="
                  stats && stats.api_online
                    ? 'tim-icons icon-link text-success'
                    : 'tim-icons icon-simple-remove text-danger'
                "
              ></i>
              {{
                stats
                  ? stats.api_online
                    ? $t("dashboard.apiOnline")
                    : $t("dashboard.apiOffline")
                  : "-"
              }}
            </h3>
          </template>
        </card>
      </div>
    </div>

    <!-- Type Distribution Chart -->
    <div class="row">
      <div class="col-12">
        <card type="chart">
          <template slot="header">
          <div class="row">
            <div class="col-sm-6" :class="isRTL ? 'text-right' : 'text-left'">
              <h5 class="card-category">{{ $t("dashboard.typeDistribution") }}</h5>
              <h2 class="card-title">{{ $t("dashboard.typeDistribution") }}</h2>
            </div>
            <div class="col-sm-6">
              <div
                class="btn-group btn-group-toggle"
                :class="isRTL ? 'float-left' : 'float-right'"
                data-toggle="buttons"
              >
                <label
                  :class="[
                    'btn btn-sm btn-primary btn-simple',
                    { active: chartMode === 'byType' },
                  ]"
                >
                  <input
                    type="radio"
                    @click="chartMode = 'byType'"
                    name="chartMode"
                    autocomplete="off"
                    :checked="chartMode === 'byType'"
                  />
                  {{ $t("dashboard.byType") }}
                </label>
                <label
                  :class="[
                    'btn btn-sm btn-primary btn-simple',
                    { active: chartMode === 'trainingTesting' },
                  ]"
                >
                  <input
                    type="radio"
                    @click="chartMode = 'trainingTesting'"
                    name="chartMode"
                    autocomplete="off"
                    :checked="chartMode === 'trainingTesting'"
                  />
                  {{ $t("dashboard.trainingTesting") }}
                </label>
              </div>
            </div>
          </div>
        </template>
        <div class="chart-area" style="min-height: 300px">
          <div v-if="typeStatsLoading" class="text-center py-5 text-muted">
            {{ $t("dashboard.loading") }}
          </div>
          <div
            v-else-if="
              !typeChartData ||
              !typeChartData.labels ||
              typeChartData.labels.length === 0
            "
            class="text-center py-5 text-muted"
          >
            {{ $t("dashboard.noData") }}
          </div>
          <bar-chart
            v-else
            :key="`type-chart-${chartMode}`"
            style="height: 100%"
            chart-id="type-distribution-chart"
            :chart-data="typeChartData"
            :gradient-colors="typeChartGradientColors"
            :gradient-stops="typeChartGradientStops"
            :extra-options="typeChartOptions"
            :use-gradient="chartMode === 'byType'"
          >
          </bar-chart>
        </div>
        </card>
      </div>
    </div>

    <!-- Recent Records -->
    <div class="row">
      <div class="col-12">
        <card :header-classes="{ 'text-right': isRTL }">
          <h4 slot="header" class="card-title">
            {{ $t("dashboard.recentRecords") }}
          </h4>
          <div v-if="viewLoading" class="text-center py-4 text-muted">
            {{ $t("dashboard.loading") }}
          </div>
          <div
            v-else-if="!recentRecords || recentRecords.length === 0"
            class="text-center py-4 text-muted"
          >
            {{ $t("dashboard.noData") }}
          </div>
          <div v-else class="table-responsive" style="max-height: 400px; overflow-y: auto;">
            <base-table
              :data="recentRecords"
              :columns="['id', 'upload_timestamp', 'T', 'type']"
              thead-classes="text-primary"
            >
            </base-table>
          </div>
        </card>
      </div>
    </div>
  </div>
</template>
<script>
import BarChart from "@/components/Charts/BarChart";
import { BaseAlert, BaseTable } from "@/components";
import * as chartConfigs from "@/components/Charts/config";
import { getStats, getTypeStats, getView } from "@/services/api";
import cache from "@/services/cache";
import config from "@/config";

export default {
  components: {
    BarChart,
    BaseAlert,
    BaseTable,
  },
  data() {
    return {
      stats: null,
      statsLoading: true,
      typeStats: null,
      typeStatsLoading: true,
      recentRecords: [],
      viewLoading: true,
      chartMode: "byType",
      typeChartGradientColors: config.colors.primaryGradient,
      typeChartGradientStops: [1, 0.4, 0],
    };
  },
  computed: {
    isRTL() {
      return this.$rtl && this.$rtl.isRTL;
    },
    typeChartOptions() {
      // Get base options without hardcoded min/max
      const baseYAxis = chartConfigs.barChartOptions.scales.yAxes[0];
      const { suggestedMin, suggestedMax, ...baseTicks } = baseYAxis.ticks || {};
      
      return {
        ...chartConfigs.barChartOptions,
        legend:
          this.chartMode === "trainingTesting"
            ? { display: true, position: "top" }
            : chartConfigs.basicOptions.legend,
        scales: {
          yAxes: [
            {
              gridLines: baseYAxis.gridLines || {},
              ticks: {
                ...baseTicks,
                beginAtZero: true,
                padding: 20,
                fontColor: "#9e9e9e",
              },
            },
          ],
          xAxes: chartConfigs.barChartOptions.scales.xAxes,
        },
      };
    },
    typeChartData() {
      if (!this.typeStats) return null;
      const dist = this.typeStats.type_distribution || {};
      const typeTraining = this.typeStats.type_training || {};
      const typeTesting = this.typeStats.type_testing || {};
      const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
      const labels = entries.map((e) => e[0]);
      const values = entries.map((e) => e[1]);

      if (this.chartMode === "byType") {
        const colors = this.generateColors(labels.length);
        return {
          labels,
          datasets: [
            {
              label: "Count",
              data: values,
              backgroundColor: colors,
              borderColor: colors.map((c) => c.replace(/0\.8\)/, "1)")),
              borderWidth: 2,
            },
          ],
        };
      }
      const trainingData = labels.map((l) => typeTraining[l] || 0);
      const testingData = labels.map((l) => typeTesting[l] || 0);
      return {
        labels,
        datasets: [
          {
            label: "Training",
            data: trainingData,
            backgroundColor: "rgba(29, 140, 248, 0.8)",
            borderColor: "rgba(29, 140, 248, 1)",
            borderWidth: 2,
          },
          {
            label: "Testing",
            data: testingData,
            backgroundColor: "rgba(255, 152, 0, 0.8)",
            borderColor: "rgba(255, 152, 0, 1)",
            borderWidth: 2,
          },
        ],
      };
    },
  },
  methods: {
    generateColors(n) {
      const base = [
        "rgba(66, 184, 131, 0.8)",  // Green
        "rgba(29, 140, 248, 0.8)",  // Blue
        "rgba(253, 93, 147, 0.8)",  // Pink
        "rgba(255, 152, 0, 0.8)",   // Orange
        "rgba(0, 214, 180, 0.8)",   // Teal
        "rgba(156, 39, 176, 0.8)",  // Purple (replaced gray)
      ];
      const out = [];
      for (let i = 0; i < n; i++) out.push(base[i % base.length]);
      return out;
    },
    async fetchDiverseRecords(totalRecords) {
      if (totalRecords === 0) {
        return { data: [] };
      }

      // Calculate offsets distributed across the dataset
      // Fetch from ~20 different points across the entire dataset
      const numSamples = 20;
      const chunkSize = 50; // Smaller chunks from each offset
      const offsets = [];
      
      // Generate offsets distributed across the dataset
      for (let i = 0; i < numSamples; i++) {
        const offset = Math.floor((totalRecords * i) / numSamples);
        if (offset < totalRecords) {
          offsets.push(offset);
        }
      }
      
      // Remove duplicates and limit to reasonable number
      const uniqueOffsets = [...new Set(offsets)].slice(0, 20);
      
      console.log(`Fetching ${chunkSize} records from ${uniqueOffsets.length} offsets across ${totalRecords} total records`);
      
      // Fetch all chunks in parallel
      const fetchPromises = uniqueOffsets.map(offset => 
        getView(chunkSize, offset, true).catch(() => ({ data: [] }))
      );
      
      const results = await Promise.all(fetchPromises);
      
      // Combine all results
      const allData = [];
      results.forEach((res) => {
        if (res && res.data && Array.isArray(res.data)) {
          allData.push(...res.data);
        }
      });
      
      const combined = { data: allData };
      // Cache the combined result
      cache.set("/api/view-multi-offset", combined);
      return combined;
    },
    sampleRecordsByType(records, samplesPerType = 8) {
      if (!records || records.length === 0) {
        return [];
      }

      // Map records to the format we need - extract type more thoroughly
      const mappedRecords = records.map((row) => {
        let typeValue = "-";
        
        if (row.data && typeof row.data === 'object') {
          // Try multiple field names (case-insensitive)
          const possibleFields = ['type', 'label', 'category', 'class', 'attack_cat', 'attack_cat_', 'Attack_cat'];
          for (const field of possibleFields) {
            // Try exact match
            if (row.data[field]) {
              typeValue = String(row.data[field]).trim();
              break;
            }
            // Try case-insensitive match
            const lowerField = field.toLowerCase();
            for (const key in row.data) {
              if (key.toLowerCase() === lowerField) {
                typeValue = String(row.data[key]).trim();
                break;
              }
            }
            if (typeValue !== "-") break;
          }
        }
        
        return {
          id: row.id,
          upload_timestamp: row.upload_timestamp || "-",
          t: row.T || "-",
          type: typeValue || "-",
          originalRow: row, // Keep reference for sampling
        };
      });
      
      // Debug: Check what fields exist in first few records
      if (records.length > 0 && records[0].data) {
        console.log("Sample record data keys:", Object.keys(records[0].data));
        console.log("Sample record data:", records[0].data);
      }

      // Group by type
      const byType = {};
      mappedRecords.forEach((record) => {
        const type = record.type;
        if (!byType[type]) {
          byType[type] = [];
        }
        byType[type].push(record);
      });

      const typeKeys = Object.keys(byType);
      
      // Log types found for debugging
      console.log(`Found ${typeKeys.length} types in fetched data:`, typeKeys);
      console.log(`Type distribution:`, Object.keys(byType).map(type => ({
        type,
        count: byType[type].length
      })));

      // Sample from each type - prioritize non-"normal" types first
      const sampled = [];
      
      // Sort types: non-"normal" types first
      const sortedTypeKeys = typeKeys.sort((a, b) => {
        if (a.toLowerCase() === "normal") return 1;
        if (b.toLowerCase() === "normal") return -1;
        return a.localeCompare(b);
      });
      
      sortedTypeKeys.forEach((type) => {
        const recordsOfType = [...byType[type]]; // Copy array
        
        // Shuffle to get diverse samples
        for (let i = recordsOfType.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [recordsOfType[i], recordsOfType[j]] = [recordsOfType[j], recordsOfType[i]];
        }
        
        const sampleSize = Math.min(samplesPerType, recordsOfType.length);
        // Take N random records from each type
        sampled.push(...recordsOfType.slice(0, sampleSize));
      });

      // Log what we sampled
      const sampledTypes = {};
      sampled.forEach(r => {
        sampledTypes[r.type] = (sampledTypes[r.type] || 0) + 1;
      });
      console.log(`Sampled ${sampled.length} records with types:`, sampledTypes);

      // Sort by type first (to group by type), then by ID
      const sorted = sampled.sort((a, b) => {
        if (a.type !== b.type) {
          return a.type.localeCompare(b.type);
        }
        return a.id - b.id;
      });
      
      // Limit to 100 records max
      return sorted.slice(0, 100);
    },
    async loadData() {
      // Check cache first for instant display when navigating back
      const cachedStats = cache.get("/api/stats");
      const cachedTypeStats = cache.get("/api/type-stats");
      // Use composite cache key for multi-offset fetch
      const cachedViewKey = `/api/view-multi-offset`;
      const cachedView = cache.get(cachedViewKey);

      // Show cached data immediately if available (instant load when navigating back)
      if (cachedStats) {
        this.stats = cachedStats;
        this.statsLoading = false;
      } else {
        this.statsLoading = true;
      }

      if (cachedTypeStats) {
        this.typeStats = cachedTypeStats;
        this.typeStatsLoading = false;
      } else {
        this.typeStatsLoading = true;
      }

      if (cachedView && cachedView.data && Array.isArray(cachedView.data) && cachedView.data.length > 0) {
        const viewData = cachedView.data;
        this.recentRecords = this.sampleRecordsByType(viewData, 15);
        this.viewLoading = false;
      } else {
        this.viewLoading = true;
      }

      // Always fetch fresh data in background (updates cache and UI with latest data)
      try {
        // Get stats first to know total records
        const statsRes = await getStats(true).catch(() => ({
          api_online: false,
          total_records: 0,
          training_records: 0,
          testing_records: 0,
          training_percentage: 0,
          testing_percentage: 0,
        }));
        
        // Now fetch type stats and view data
        const [typeRes, viewRes] = await Promise.all([
          getTypeStats(true).catch(() => null),
          // Fetch from many distributed offsets across the entire dataset
          this.fetchDiverseRecords(statsRes.total_records || 0).catch((err) => {
            console.error("Error fetching diverse view data:", err);
            return { data: [] };
          }),
        ]);

        // Update with fresh data (cache is automatically updated by API service)
        this.stats = statsRes;
        this.typeStats = typeRes;
        
        // Ensure we have valid data array
        const viewData = (viewRes && viewRes.data && Array.isArray(viewRes.data)) ? viewRes.data : [];
        // Sample records from each type for diverse display (15 per type, up to ~100 total)
        this.recentRecords = this.sampleRecordsByType(viewData, 15);
        
        // Debug: log sampling info
        console.log(`Loaded ${viewData.length} records, sampled ${this.recentRecords.length} records across types`);
        if (viewData.length === 0) {
          console.log("No data returned from /api/view - response:", viewRes);
        }
      } finally {
        this.statsLoading = false;
        this.typeStatsLoading = false;
        this.viewLoading = false;
      }
    },
  },
  mounted() {
    this.loadData();
  },
  activated() {
    // This fires when component is activated from keep-alive
    // If we already have data, just refresh in background
    // If no data, do full load
    if (this.stats || this.typeStats) {
      // We have data, just refresh in background silently
      this.loadData();
    } else {
      // No data, do full load
      this.loadData();
    }
  },
};
</script>
<style></style>
