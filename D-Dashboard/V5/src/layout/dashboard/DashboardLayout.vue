<template>
  <div class="wrapper">
    <side-bar :backgroundColor="sidebarBgColor">
      <template slot="links">
        <sidebar-link
          to="/dashboard"
          :name="$t('sidebar.dashboard')"
          icon="tim-icons icon-chart-pie-36"
        />
        <sidebar-link
          to="/icons"
          :name="$t('sidebar.icons')"
          icon="tim-icons icon-atom"
        />
        <sidebar-link
          to="/maps"
          :name="$t('sidebar.maps')"
          icon="tim-icons icon-pin"
        />
        <sidebar-link
          to="/notifications"
          :name="$t('sidebar.notifications')"
          icon="tim-icons icon-bell-55"
        />
        <sidebar-link
          to="/profile"
          :name="$t('sidebar.userProfile')"
          icon="tim-icons icon-single-02"
        />
        <sidebar-link
          to="/table-list"
          :name="$t('sidebar.tableList')"
          icon="tim-icons icon-puzzle-10"
        />
        <sidebar-link
          to="/typography"
          :name="$t('sidebar.typography')"
          icon="tim-icons icon-align-center"
        />
        <sidebar-link
          to="/dashboard?enableRTL=true"
          :name="$t('sidebar.rtlSupport')"
          icon="tim-icons icon-world"
        />
      </template>
    </side-bar>
    <div class="main-panel" :data="sidebarBgColor">
      <top-navbar></top-navbar>

      <dashboard-content @click.native="toggleSidebar"> </dashboard-content>

      <content-footer></content-footer>
    </div>
    <fixed-plugin
      :bg-color="sidebarBgColor"
      :is-light-mode="isLightMode"
      @update:bgColor="sidebarBgColor = $event"
      @update:theme="isLightMode = $event"
    />
  </div>
</template>
<style lang="scss"></style>
<script>
import TopNavbar from "./TopNavbar.vue";
import ContentFooter from "./ContentFooter.vue";
import DashboardContent from "./Content.vue";
import MobileMenu from "./MobileMenu";
import FixedPlugin from "@/components/FixedPlugin";

export default {
  components: {
    TopNavbar,
    ContentFooter,
    DashboardContent,
    FixedPlugin,
  },
  data() {
    return {
      sidebarBgColor: "blue",
      isLightMode: false,
    };
  },
  watch: {
    isLightMode(val) {
      if (val) {
        document.body.classList.add("white-content");
      } else {
        document.body.classList.remove("white-content");
      }
    },
  },
  mounted() {
    if (this.isLightMode) {
      document.body.classList.add("white-content");
    }
  },
  beforeDestroy() {
    document.body.classList.remove("white-content");
  },
  methods: {
    toggleSidebar() {
      if (this.$sidebar.showSidebar) {
        this.$sidebar.displaySidebar(false);
      }
    },
  },
};
</script>
