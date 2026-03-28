google.maps.__gjsload__('overlay', function(_) {
    var nI = function(a) {
            a.nh = a.nh || new kBa;
            return a.nh
        },
        nBa = function(a, b) {
            function c() {
                e ? .mh.ri()
            }
            const d = nI(a);
            let e = d.mh;
            e || (e = d.mh = new lBa(a));
            _.Ob(d.Wh || [], _.Jn);
            var f = d.oh = d.oh || new _.cqa;
            const g = b.__gm;
            f.bindTo("zoom", g);
            f.bindTo("offset", g);
            f.bindTo("center", g, "projectionCenterQ");
            f.bindTo("projection", b);
            f.bindTo("projectionTopLeft", g);
            f = d.ph = d.ph || new mBa(f);
            f.bindTo("zoom", g);
            f.bindTo("offset", g);
            f.bindTo("projection", b);
            f.bindTo("projectionTopLeft", g);
            a.bindTo("projection", f, "outProjection");
            a.bindTo("panes", g);
            d.Wh = [_.Hn(a, "panes_changed", c), _.Hn(g, "zoom_changed", c), _.Hn(g, "offset_changed", c), _.Hn(b, "projection_changed", c), _.Hn(g, "projectioncenterq_changed", c)];
            c();
            b instanceof _.po ? _.N(b, 148440) : b instanceof _.$o && _.N(b, 181451)
        },
        oBa = function(a) {
            const b = nI(a);
            var c = b.oh;
            c && c.unbindAll();
            (c = b.ph) && c.unbindAll();
            a.unbindAll();
            a.set("panes", null);
            a.set("projection", null);
            b.Wh && b.Wh.forEach(d => {
                _.Jn(d)
            });
            b.Wh = null;
            b.mh && (_.pq(b.mh.mh), b.mh = null)
        },
        tBa = function(a) {
            if (a) {
                var b = a.getMap();
                if (pBa(a) !==
                    b && b && b instanceof _.po) {
                    const c = b.__gm;
                    c.overlayLayer ? a.__gmop = new qBa(b, a, c.overlayLayer) : c.nh.then(({
                        Mh: d
                    }) => {
                        const e = new rBa(b, d);
                        d.Cj(e);
                        c.overlayLayer = e;
                        sBa(a);
                        tBa(a)
                    })
                }
            }
        },
        sBa = function(a) {
            if (a) {
                var b = a.__gmop;
                b && (a.__gmop = null, b.overlay.unbindAll(), b.overlay.set("panes", null), b.overlay.set("projection", null), b.overlayLayer.hp(b), b.mh && (b.mh = !1, b.overlay.onRemove ? b.overlay.onRemove() : b.overlay.remove()))
            }
        },
        pBa = function(a) {
            return (a = a.__gmop) ? a.map : null
        },
        uBa = function(a, b) {
            a.overlay.get("projection") !==
                b && (a.overlay.bindTo("panes", a.map.__gm), a.overlay.set("projection", b))
        },
        mBa = class extends _.$n {
            constructor(a) {
                super();
                this.projection = a
            }
            changed(a) {
                a !== "outProjection" && (a = !!(this.get("offset") && this.get("projectionTopLeft") && this.get("projection") && _.vm(this.get("zoom"))), a === !this.get("outProjection") && this.set("outProjection", a ? this.projection : null))
            }
        };
    var kBa = class {},
        lBa = class extends _.$n {
            constructor(a) {
                super();
                this.mh = new _.oq(() => {
                    const b = a.nh;
                    if (a.getPanes()) {
                        if (a.getProjection()) {
                            if (!b ? .add && a.onAdd) a.onAdd();
                            b.add = !0;
                            a.draw ? .()
                        }
                    } else {
                        if (b ? .add)
                            if (a.onRemove) a.onRemove();
                            else a.remove();
                        b.add = !1
                    }
                }, 0)
            }
        };
    var qBa = class {
            constructor(a, b, c) {
                this.map = a;
                this.overlay = b;
                this.overlayLayer = c;
                this.mh = !1;
                _.N(this.map, 148440);
                c.Go(this)
            }
            draw() {
                this.mh || (this.mh = !0, this.overlay.onAdd && this.overlay.onAdd());
                this.overlay.draw && this.overlay.draw()
            }
        },
        rBa = class {
            constructor(a, b) {
                this.map = a;
                this.Mh = b;
                this.mh = null;
                this.nh = []
            }
            dispose() {}
            ni(a, b, c, d, e, f, g, h) {
                const k = this.mh = this.mh || new _.KB(this.map, this.Mh, () => {});
                k.ni(a, b, c, d, e, f, g, h);
                for (const n of this.nh) uBa(n, k), n.draw()
            }
            Go(a) {
                this.nh.push(a);
                this.mh && uBa(a, this.mh);
                this.Mh.refresh()
            }
            hp(a) {
                _.Yb(this.nh, a)
            }
        };
    _.Sl("overlay", {
        gE: function(a) {
            if (a) {
                oBa(a);
                delete nI(a).nh;
                sBa(a);
                var b = a.getMap();
                b && (b instanceof _.po ? tBa(a) : a && (b = a.getMap(), (nI(a).nh || null) !== b && (b && nBa(a, b), nI(a).nh = b)))
            }
        },
        preventMapHitsFrom: a => {
            _.jy(a, {
                nl: ({
                    event: b
                }) => {
                    _.Zw(b.mh)
                },
                zl: b => {
                    _.Vx(b)
                },
                Or: b => {
                    _.Wx(b)
                },
                wm: b => {
                    _.Wx(b)
                },
                Ol: b => {
                    _.Xx(b)
                }
            }).Yr(!0)
        },
        preventMapHitsAndGesturesFrom: a => {
            a.addEventListener("click", _.Fn);
            a.addEventListener("contextmenu", _.Fn);
            a.addEventListener("dblclick", _.Fn);
            a.addEventListener("mousedown", _.Fn);
            a.addEventListener("mousemove",
                _.Fn);
            a.addEventListener("MSPointerDown", _.Fn);
            a.addEventListener("pointerdown", _.Fn);
            a.addEventListener("touchstart", _.Fn);
            a.addEventListener("wheel", _.Fn)
        }
    });
});