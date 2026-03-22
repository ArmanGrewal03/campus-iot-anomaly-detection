google.maps.__gjsload__('infowindow', function(_) {
    var kPa = function(a, b) {
            if (a.mh.size === 1) {
                const c = Array.from(a.mh.values())[0];
                c.fw !== b.fw && (c.set("map", null), a.mh.delete(c))
            }
            a.mh.add(b)
        },
        mPa = function(a, b) {
            var c = a.__gm;
            a = c.get("panes");
            c = c.get("innerContainer");
            b = {
                xm: a,
                Ij: _.oC.Ij(),
                qy: c,
                shouldFocus: b
            };
            return new lPa(b)
        },
        jP = function(a, b) {
            a.container.style.visibility = b ? "" : "hidden";
            b && a.shouldFocus && (a.focus(), a.shouldFocus = !1);
            b ? nPa(a) : a.xh = !1
        },
        oPa = function(a) {
            a.Gj.setAttribute("aria-labelledby", a.ph.id)
        },
        pPa = function(a) {
            const b = !!a.get("open");
            var c =
                a.get("content");
            c = b ? c : null;
            if (c === a.rh) jP(a, b && a.get("position"));
            else {
                if (a.rh) {
                    const d = a.rh.parentNode;
                    d === a.mh && d.removeChild(a.rh)
                }
                c && (a.wh = !1, a.mh.appendChild(c));
                jP(a, b && a.get("position"));
                a.rh = c;
                kP(a)
            }
        },
        lP = function(a) {
            var b = !!a.get("open"),
                c = a.get("headerContent");
            const d = !!a.get("ariaLabel"),
                e = !a.get("headerDisabled");
            b = b ? c : null;
            a.Gj.style.paddingTop = e ? "0" : "12px";
            b === a.sh ? a.oh.style.display = e ? "" : "none" : (a.sh && (c = a.sh.parentNode, c === a.ph && c.removeChild(a.sh)), b && (a.wh = !1, a.ph.appendChild(b),
                e && !d && oPa(a)), a.oh.style.display = e ? "" : "none", a.sh = b, kP(a))
        },
        kP = function(a) {
            var b = a.getSize();
            if (b) {
                var c = b.on;
                b = b.minWidth;
                a.Gj.style.maxWidth = _.Em(c.width);
                a.Gj.style.maxHeight = _.Em(c.height);
                a.Gj.style.minWidth = _.Em(b);
                a.mh.style.maxHeight = _.Bq.mh ? _.Em(c.height - 18) : _.Em(c.height - 36);
                mP(a);
                a.uh.start()
            }
        },
        qPa = function(a) {
            const b = a.get("pixelOffset") || new _.No(0, 0);
            var c = new _.No(a.Gj.offsetWidth, a.Gj.offsetHeight);
            a = -b.height + c.height + 11 + 60;
            let d = b.height + 60;
            const e = -b.width + c.width / 2 + 60;
            c = b.width +
                c.width / 2 + 60;
            b.height < 0 && (d -= b.height);
            return {
                top: a,
                bottom: d,
                left: e,
                right: c
            }
        },
        nPa = function(a) {
            !a.xh && a.get("open") && a.get("visible") && a.get("position") && (_.Wn(a, "visible"), a.xh = !0)
        },
        mP = function(a) {
            var b = a.get("position");
            if (b && a.get("pixelOffset")) {
                var c = qPa(a);
                const d = b.x - c.left,
                    e = b.y - c.top,
                    f = b.x + c.right;
                c = b.y + c.bottom;
                _.sx(a.anchor, b);
                b = a.get("zIndex");
                _.ux(a.container, _.vm(b) ? b : e + 60);
                a.set("pixelBounds", _.sp(d, e, f, c))
            }
        },
        sPa = function(a, b, c) {
            return b instanceof _.po ? new rPa(a, b, c) : new rPa(a, b)
        },
        uPa = function(a) {
            a.mh && a.aj.push(_.Tn(a.mh, "pixelposition_changed", () => {
                tPa(a)
            }))
        },
        tPa = function(a) {
            const b = a.model.get("pixelPosition") || a.mh && a.mh.get("pixelPosition");
            a.ph.set("position", b)
        },
        wPa = function(a) {
            a = a.__gm;
            a.get("IW_AUTO_CLOSER") || a.set("IW_AUTO_CLOSER", new vPa);
            return a.get("IW_AUTO_CLOSER")
        },
        vPa = class {
            constructor() {
                this.mh = new Set
            }
        };
    var lPa = class extends _.$n {
        constructor(a) {
            super();
            this.rh = this.sh = this.th = null;
            this.xh = this.wh = !1;
            this.qy = a.qy;
            this.shouldFocus = a.shouldFocus;
            this.container = document.createElement("div");
            this.container.style.cursor = "default";
            this.container.style.position = "absolute";
            this.container.style.left = this.container.style.top = "0";
            a.xm.floatPane.appendChild(this.container);
            this.anchor = document.createElement("div");
            this.container.appendChild(this.anchor);
            this.qh = document.createElement("div");
            this.anchor.appendChild(this.qh);
            this.Gj = document.createElement("div");
            this.qh.appendChild(this.Gj);
            this.Gj.setAttribute("role", "dialog");
            this.Gj.tabIndex = -1;
            this.oh = document.createElement("div");
            this.Gj.appendChild(this.oh);
            this.ph = document.createElement("div");
            this.oh.appendChild(this.ph);
            this.zh = document.createElement("div");
            this.qh.appendChild(this.zh);
            this.mh = document.createElement("div");
            this.Gj.appendChild(this.mh);
            _.TFa(this.container);
            _.nx(this.Gj, "gm-style-iw");
            _.nx(this.anchor, "gm-style-iw-a");
            _.nx(this.qh, "gm-style-iw-t");
            _.nx(this.zh, "gm-style-iw-tc");
            _.nx(this.Gj, "gm-style-iw-c");
            _.nx(this.oh, "gm-style-iw-chr");
            _.nx(this.ph, "gm-style-iw-ch");
            _.nx(this.mh, "gm-style-iw-d");
            this.ph.setAttribute("id", _.oo());
            _.Bq.mh && !_.Bq.uh && (this.Gj.style.paddingInlineEnd = "0", this.Gj.style.paddingBottom = "0", this.mh.style.overflow = "scroll");
            jP(this, !1);
            _.Pn(this.container, "mousedown", _.Fn);
            _.Pn(this.container, "mouseup", _.Fn);
            _.Pn(this.container, "mousemove", _.Fn);
            _.Pn(this.container, "pointerdown", _.Fn);
            _.Pn(this.container, "pointerup",
                _.Fn);
            _.Pn(this.container, "pointermove", _.Fn);
            _.Pn(this.container, "dblclick", _.Fn);
            _.Pn(this.container, "click", _.Fn);
            _.Pn(this.container, "touchstart", _.Fn);
            _.Pn(this.container, "touchend", _.Fn);
            _.Pn(this.container, "touchmove", _.Fn);
            _.ax(this.container, "contextmenu", this, this.Ah);
            _.ax(this.container, "wheel", this, _.Fn);
            a = new _.Jo(12, 12);
            const b = new _.No(24, 24);
            this.nh = new _.Hr({
                Jr: a,
                Vs: b,
                offset: new _.Jo(-6, -6),
                XC: !0,
                ownerElement: this.oh
            });
            this.oh.appendChild(this.nh.element);
            _.Pn(this.nh.element, "click",
                c => {
                    _.Fn(c);
                    _.Wn(this, "closeclick");
                    this.set("open", !1)
                });
            this.uh = new _.oq(() => {
                !this.wh && this.get("content") && this.get("visible") && (_.Wn(this, "domready"), this.wh = !0)
            }, 0);
            this.yh = _.Pn(this.container, "keydown", c => {
                c.key !== "Escape" && c.key !== "Esc" || !this.Gj.contains(document.activeElement) || (c.stopPropagation(), _.Wn(this, "closeclick"), this.set("open", !1))
            })
        }
        ariaLabel_changed() {
            const a = this.get("ariaLabel");
            a ? this.Gj.setAttribute("aria-label", a) : (this.Gj.removeAttribute("aria-label"), this.get("headerDisabled") ||
                oPa(this))
        }
        open_changed() {
            pPa(this);
            lP(this)
        }
        headerContent_changed() {
            lP(this)
        }
        headerDisabled_changed() {
            lP(this)
        }
        content_changed() {
            pPa(this)
        }
        pendingFocus_changed() {
            this.get("pendingFocus") && (this.get("open") && this.get("visible") && this.get("position") ? _.Iq(this.Gj, !0) : console.warn("Setting focus on InfoWindow was ignored. This is most likely due to InfoWindow not being visible yet."), this.set("pendingFocus", !1))
        }
        dispose() {
            setTimeout(() => {
                document.activeElement && document.activeElement !== document.body ||
                    (this.th && this.th !== document.body ? _.Iq(this.th, !0) || _.Iq(this.qy, !0) : _.Iq(this.qy, !0))
            });
            this.yh && _.Jn(this.yh);
            this.container.parentNode.removeChild(this.container);
            this.uh.stop();
            this.uh.dispose()
        }
        getSize() {
            var a = this.get("layoutPixelBounds"),
                b = this.get("pixelOffset");
            const c = this.get("maxWidth") || 648,
                d = this.get("minWidth") || 0;
            if (!b) return null;
            a ? (b = a.maxY - a.minY - (11 + -b.height), a = a.maxX - a.minX - 6, a >= 240 && (a -= 120), b >= 240 && (b -= 120)) : (a = 648, b = 654);
            a = Math.min(a, c);
            a = Math.max(d, a);
            a = Math.max(0, a);
            b = Math.max(0,
                b);
            return {
                on: new _.No(a, b),
                minWidth: d
            }
        }
        pixelOffset_changed() {
            const a = this.get("pixelOffset") || new _.No(0, 0);
            this.qh.style.right = _.Em(-a.width);
            this.qh.style.bottom = _.Em(-a.height + 11);
            kP(this)
        }
        layoutPixelBounds_changed() {
            kP(this)
        }
        position_changed() {
            this.get("position") ? (mP(this), jP(this, !!this.get("open"))) : jP(this, !1)
        }
        zIndex_changed() {
            mP(this)
        }
        visible_changed() {
            this.container.style.display = this.get("visible") ? "" : "none";
            this.uh.start();
            if (this.get("visible")) {
                const a = this.nh.element.style.display;
                this.nh.element.style.display = "none";
                this.nh.element.getBoundingClientRect();
                this.nh.element.style.display = a;
                nPa(this)
            } else this.xh = !1
        }
        Ah(a) {
            let b = !1;
            const c = this.get("content");
            let d = a.target;
            for (; !b && d;) b = d === c, d = d.parentNode;
            b ? _.Cn(a) : _.En(a)
        }
        focus() {
            this.th = document.activeElement;
            let a;
            _.Bq.wh && (a = this.mh.getBoundingClientRect());
            if (this.get("disableAutoPan")) _.Iq(this.Gj, !0);
            else {
                var b = _.yx(this.mh);
                if (b.length) {
                    b = b[0];
                    a = a || this.mh.getBoundingClientRect();
                    var c = b.getBoundingClientRect();
                    _.Iq(c.bottom <=
                        a.bottom && c.right <= a.right ? b : this.Gj, !0)
                } else _.Iq(this.nh.element, !0)
            }
        }
    };
    var rPa = class {
        constructor(a, b, c) {
            this.model = a;
            this.isOpen = !0;
            this.mh = this.oh = this.Mh = null;
            this.aj = [];
            var d = a.get("shouldFocus");
            this.ph = mPa(b, d);
            const e = b.__gm;
            (d = b instanceof _.po) && c ? c.then(h => {
                this.isOpen && (this.Mh = h, this.mh = new _.$M(k => {
                    this.oh = new _.KB(b, h, k, () => {});
                    h.Cj(this.oh);
                    return this.oh
                }), this.mh.bindTo("latLngPosition", a, "position"), uPa(this))
            }) : (this.mh = new _.$M, this.mh.bindTo("latLngPosition", a, "position"), this.mh.bindTo("center", e, "projectionCenterQ"), this.mh.bindTo("zoom", e), this.mh.bindTo("offset",
                e), this.mh.bindTo("projection", b), this.mh.bindTo("focus", b, "position"), uPa(this));
            this.qh = d ? a.infoWindow.get("logAsInternal") ? 148284 : 148285 : null;
            const f = new _.PM(["scale"], "visible", h => h == null || h >= .3);
            this.mh && f.bindTo("scale", this.mh);
            const g = this.ph;
            g.set("logAsInternal", !!a.infoWindow.get("logAsInternal"));
            g.bindTo("ariaLabel", a);
            g.bindTo("zIndex", a);
            g.bindTo("layoutPixelBounds", e, "pixelBounds");
            g.bindTo("disableAutoPan", a);
            g.bindTo("pendingFocus", a);
            g.bindTo("maxWidth", a);
            g.bindTo("minWidth",
                a);
            g.bindTo("content", a);
            g.bindTo("headerContent", a);
            g.bindTo("headerDisabled", a);
            g.bindTo("pixelOffset", a);
            g.bindTo("visible", f);
            this.nh = new _.oq(() => {
                if (b instanceof _.po)
                    if (this.Mh) {
                        var h = a.get("position");
                        h && (0, _.Apa.AG)(b, this.Mh, new _.wo(h), qPa(g))
                    } else c.then(() => {
                        this.nh.start()
                    });
                else(h = g.get("pixelBounds")) ? _.Wn(e, "pantobounds", h) : this.nh.start()
            }, 150);
            if (d) {
                let h = null;
                this.aj.push(_.Tn(a, "position_changed", () => {
                    const k = a.get("position");
                    !k || a.get("disableAutoPan") || k.equals(h) || (this.nh.start(),
                        h = k)
                }))
            } else a.get("disableAutoPan") || this.nh.start();
            g.set("open", !0);
            this.aj.push(_.Hn(g, "domready", () => {
                a.trigger("domready")
            }));
            this.aj.push(_.Hn(g, "visible", () => {
                a.trigger("visible")
            }));
            this.aj.push(_.Hn(g, "closeclick", () => {
                a.close();
                a.trigger("closeclick")
            }));
            this.aj.push(_.Tn(a, "pixelposition_changed", () => {
                tPa(this)
            }));
            this.qh && _.N(b, this.qh)
        }
        close() {
            if (this.isOpen) {
                this.isOpen = !1;
                this.model.trigger("close");
                for (var a of this.aj) _.Jn(a);
                this.aj.length = 0;
                this.nh.stop();
                this.nh.dispose();
                this.Mh &&
                    this.oh && this.Mh.Ql(this.oh);
                a = this.ph;
                a.unbindAll();
                a.set("open", !1);
                a.dispose();
                this.mh && this.mh.unbindAll()
            }
        }
    };
    _.Sl("infowindow", {
        aJ: function(a) {
            let b = null;
            _.Tn(a, "map_changed", function d() {
                const e = a.get("map");
                b && (b.oE.mh.delete(a), b.JM.close(), b = null);
                if (e) {
                    const f = e.__gm;
                    f.get("panes") ? f.get("innerContainer") ? (b = {
                        JM: sPa(a, e, e instanceof _.po ? f.nh.then(({
                            Mh: g
                        }) => g) : void 0),
                        oE: wPa(e)
                    }, kPa(b.oE, a)) : _.Sn(f, "innercontainer_changed", d) : _.Sn(f, "panes_changed", d)
                }
            })
        }
    });
});