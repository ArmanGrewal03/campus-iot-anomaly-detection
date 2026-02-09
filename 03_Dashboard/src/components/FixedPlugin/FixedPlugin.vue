<template>
  <div class="fixed-plugin" v-click-outside="closeDropDown">
    <div class="dropdown" :class="{ show: isOpen }">
      <a href="#" class="dropdown-toggle" @click.prevent="toggleDropDown">
        <i class="fa fa-cog fa-2x"></i>
      </a>
      <ul class="dropdown-menu">
        <li class="header-title">SIDEBAR BACKGROUND</li>
        <li class="adjustments-line">
          <a href="javascript:void(0)" class="switch-trigger background-color">
            <div class="badge-colors text-center">
              <span
                class="badge filter badge-primary"
                :class="{ active: bgColor === 'primary' }"
                data-color="primary"
                @click="onBgColorClick('primary')"
              ></span>
              <span
                class="badge filter badge-info"
                :class="{ active: bgColor === 'blue' }"
                data-color="blue"
                @click="onBgColorClick('blue')"
              ></span>
              <span
                class="badge filter badge-success"
                :class="{ active: bgColor === 'green' }"
                data-color="green"
                @click="onBgColorClick('green')"
              ></span>
            </div>
            <div class="clearfix"></div>
          </a>
        </li>
        <li class="adjustments-line text-center color-change">
          <span class="color-label">LIGHT MODE</span>
          <span
            class="badge light-badge mr-2"
            :class="{ active: isLightMode }"
            @click="onThemeClick(true)"
          ></span>
          <span
            class="badge dark-badge ml-2"
            :class="{ active: !isLightMode }"
            @click="onThemeClick(false)"
          ></span>
          <span class="color-label">DARK MODE</span>
        </li>
        <li class="button-container">
          <a
            href="https://www.creative-tim.com/product/black-dashboard"
            target="_blank"
            rel="noopener"
            class="btn btn-primary btn-block btn-round"
          >
            Download Now
          </a>
          <a
            href="https://demos.creative-tim.com/black-dashboard/docs/1.0/getting-started/introduction.html"
            target="_blank"
            rel="noopener"
            class="btn btn-default btn-block btn-round"
          >
            Documentation
          </a>
        </li>
      </ul>
    </div>
  </div>
</template>
<script>
export default {
  name: "FixedPlugin",
  props: {
    bgColor: {
      type: String,
      default: "primary",
      validator: (v) => ["primary", "blue", "green"].includes(v),
    },
    isLightMode: {
      type: Boolean,
      default: false,
    },
  },
  data() {
    return {
      isOpen: false,
    };
  },
  methods: {
    toggleDropDown() {
      this.isOpen = !this.isOpen;
    },
    closeDropDown() {
      this.isOpen = false;
    },
    onBgColorClick(color) {
      this.$emit("update:bgColor", color);
    },
    onThemeClick(light) {
      this.$emit("update:theme", light);
    },
  },
};
</script>
