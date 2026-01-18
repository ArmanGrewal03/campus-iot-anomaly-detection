import { Bar, mixins } from "vue-chartjs";

export default {
  name: "bar-chart",
  extends: Bar,
  mixins: [mixins.reactiveProp],
  props: {
    extraOptions: Object,
    useGradient: {
      type: Boolean,
      default: true,
    },
    gradientColors: {
      type: Array,
      default: () => [
        "rgba(72,72,176,0.2)",
        "rgba(72,72,176,0.0)",
        "rgba(119,52,169,0)",
      ],
      validator: (val) => {
        return val.length > 2;
      },
    },
    gradientStops: {
      type: Array,
      default: () => [1, 0.4, 0],
      validator: (val) => {
        return val.length > 2;
      },
    },
  },
  data() {
    return {
      ctx: null,
    };
  },
  methods: {
    updateGradients(chartData) {
      if (!chartData || !this.useGradient) return;
      const ctx =
        this.ctx || document.getElementById(this.chartId).getContext("2d");

      // Helper to extract RGB values from rgba/rgb string
      const extractRGB = (color) => {
        if (typeof color !== 'string') return null;
        const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        if (rgbaMatch) {
          return {
            r: parseInt(rgbaMatch[1]),
            g: parseInt(rgbaMatch[2]),
            b: parseInt(rgbaMatch[3])
          };
        }
        return null;
      };

      // Helper to ensure semi-transparent alpha
      const ensureAlpha = (color, targetAlpha = 0.8) => {
        if (typeof color !== 'string') return color;
        const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
        if (rgbaMatch) {
          const alpha = rgbaMatch[4] ? parseFloat(rgbaMatch[4]) : 1;
          const finalAlpha = alpha > 0.9 || alpha < 0.1 ? targetAlpha : alpha;
          return `rgba(${rgbaMatch[1]}, ${rgbaMatch[2]}, ${rgbaMatch[3]}, ${finalAlpha})`;
        }
        return color;
      };

      chartData.datasets.forEach((set) => {
        if (Array.isArray(set.backgroundColor)) {
          // For per-bar colors, create individual gradients for each bar
          // Each gradient starts from the bar's color (semi-transparent) and fades to transparent
          set.backgroundColor = set.backgroundColor.map((barColor) => {
            const rgb = extractRGB(barColor);
            if (!rgb) return barColor; // Fallback if color parsing fails

            // Create a gradient for this specific bar color
            const gradient = ctx.createLinearGradient(0, 230, 0, 50);
            // Start with the bar color at ~0.3 alpha (very transparent)
            gradient.addColorStop(this.gradientStops[0], `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.3)`);
            // Fade to ~0.1 alpha
            gradient.addColorStop(this.gradientStops[1], `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.1)`);
            // Fade to transparent
            gradient.addColorStop(this.gradientStops[2], `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0)`);
            
            return gradient;
          });
        } else {
          // For single color, use the configured gradient colors
          const gradientStroke = ctx.createLinearGradient(0, 230, 0, 50);
          gradientStroke.addColorStop(
            this.gradientStops[0],
            ensureAlpha(this.gradientColors[0], 0.3)
          );
          gradientStroke.addColorStop(
            this.gradientStops[1],
            ensureAlpha(this.gradientColors[1], 0.1)
          );
          gradientStroke.addColorStop(
            this.gradientStops[2],
            ensureAlpha(this.gradientColors[2], 0)
          );
          set.backgroundColor = gradientStroke;
        }
      });
    },
  },
  mounted() {
    this.$watch(
      "chartData",
      (newVal, oldVal) => {
        this.updateGradients(newVal);
        if (!oldVal) {
          this.renderChart(this.chartData, this.extraOptions);
        }
      },
      { immediate: true }
    );
  },
};
