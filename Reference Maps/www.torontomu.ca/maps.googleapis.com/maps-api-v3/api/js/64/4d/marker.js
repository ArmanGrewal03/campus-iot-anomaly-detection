google.maps.__gjsload__('marker', function(_) {
    var zVa = function(a, b) {
            const c = _.Ka(b);
            a.mh.set(c, b);
            a.nh.ri()
        },
        AVa = function(a, b) {
            if (a.nh.has(b)) {
                _.Ln(b, "UPDATE_BASEMAP_COLLISION");
                _.Ln(b, "UPDATE_MARKER_COLLISION");
                _.Ln(b, "REMOVE_COLLISION");
                a.nh.delete(b);
                var c = a.oh;
                const d = _.Ka(b);
                c.mh.has(d) && (c.mh.delete(d), b.Wo = null, c.nh.ri());
                _.Cca(a.mh, b)
            }
        },
        BVa = function(a, b) {
            a.nh.has(b) || (a.nh.add(b), _.Hn(b, "UPDATE_BASEMAP_COLLISION", () => {
                a.ph.add(b);
                a.qh.gq()
            }), _.Hn(b, "UPDATE_MARKER_COLLISION", () => {
                a.qh.gq()
            }), _.Hn(b, "REMOVE_COLLISION", () => {
                AVa(a, b)
            }), zVa(a.oh,
                b), _.Bca(a.mh, b))
        },
        CVa = function(a) {
            return `${_.Km(a)}-internal-${a}`
        },
        DVa = function(a, b) {
            b = (a = a.__e3_) && a[b];
            return !!b && Object.values(b).some(c => c.ZC)
        },
        EVa = function(a, b, c) {
            return new _.Gn(a, `${b}${"_removed"}`, c, 0, !1)
        },
        FVa = function(a, b, c) {
            return new _.Gn(a, `${b}${"_added"}`, c, 0, !1)
        },
        BR = function(a) {
            return a ? _.ar.has(a) ? _.ar.get(a) : a instanceof ShadowRoot ? BR(a.host) : BR(a.parentNode) : null
        },
        GVa = function(a) {
            var b = 1;
            return () => {
                --b || a()
            }
        },
        CR = function(a) {
            return a instanceof _.iC
        },
        DR = function(a) {
            return CR(a) ?
                a.getSize() : a.size
        },
        HVa = function(a, b) {
            if (!(a && b && a.isConnected && b.isConnected)) return !1;
            a = a.getBoundingClientRect();
            b = b.getBoundingClientRect();
            return b.x + b.width < a.x - 0 || b.x > a.x + a.width + 0 || b.y + b.height < a.y - 0 || b.y > a.y + a.height + 0 ? !1 : !0
        },
        IVa = function(a, b) {
            _.VI().mh.load(new _.uL(a), c => {
                b(c && c.size)
            })
        },
        JVa = function(a, b) {
            a = a.getBoundingClientRect();
            b = b instanceof Element ? b.getBoundingClientRect() : a;
            return {
                offset: new _.Jo(b.x - a.x, b.y - a.y),
                size: new _.No(b.width, b.height)
            }
        },
        ER = function(a) {
            a = new DOMMatrixReadOnly(a.transform);
            return {
                offsetX: a.m41,
                offsetY: a.m42
            }
        },
        FR = function(a) {
            const b = window.devicePixelRatio || 1;
            return Math.round(a * b) / b
        },
        KVa = function(a, {
            clientX: b,
            clientY: c
        }) {
            const {
                height: d,
                left: e,
                top: f,
                width: g
            } = a.getBoundingClientRect();
            return {
                Sh: FR(b - (e + g / 2)),
                Th: FR(c - (f + d / 2))
            }
        },
        LVa = function(a, b) {
            if (!a || !b) return null;
            a = a.getProjection();
            return _.Cx(b, a)
        },
        MVa = function(a, b) {
            const c = _.ZL(a);
            if (!b || !c) return !1;
            a = Math.abs(c.clientX - b.clientX);
            b = Math.abs(c.clientY - b.clientY);
            return a * a + b * b >= 4
        },
        NVa = function(a, b) {
            const c = [];
            c.push("@-webkit-keyframes ", b, " {\n");
            _.Ob(a.frames, d => {
                c.push(d.time * 100 + "% { ");
                c.push("-webkit-transform: translate3d(" + d.translate[0] + "px,", d.translate[1] + "px,0); ");
                c.push("-webkit-animation-timing-function: ", d.sn, "; ");
                c.push("}\n")
            });
            c.push("}\n");
            return c.join("")
        },
        OVa = function(a, b) {
            for (let c = 0; c < a.frames.length - 1; c++) {
                const d = a.frames[c + 1];
                if (b >= a.frames[c].time && b < d.time) return c
            }
            return a.frames.length - 1
        },
        PVa = function(a) {
            if (a.mh) return a.mh;
            a.mh = "_gm" + Math.round(Math.random() * 1E4);
            var b =
                NVa(a, a.mh);
            GR || (GR = _.Bl("style"), GR.type = "text/css", document.querySelectorAll("HEAD")[0].appendChild(GR));
            b = GR.textContent + b;
            b = _.Hl(b);
            GR.textContent = _.Wi(new _.Vi(b));
            return a.mh
        },
        QVa = function(a) {
            a = a.get("collisionBehavior");
            return a === "REQUIRED_AND_HIDES_OPTIONAL" || a === "OPTIONAL_AND_HIDES_LOWER_PRIORITY"
        },
        RVa = function(a, b, c = !1) {
            if (!b.get("internalMarker")) {
                _.N(a, 149055);
                c ? _.N(a, 149060) : a instanceof _.po ? _.N(a, 149057) : a instanceof _.$o && (_.N(a, 149059), a.get("standAlone") && _.N(a, 149058));
                c = a.get("styles") || [];
                Array.isArray(c) && c.some(e => "stylers" in e) && _.N(a, 174113);
                QVa(b) && _.N(a, 149062);
                b.get("anchorPoint") && _.N(a, 149064);
                c = b.get("animation");
                c === 1 && _.N(a, 149065);
                c === 2 && _.N(a, 149066);
                b.get("clickable") === !1 && (_.N(a, 149091), b.get("title") && _.N(a, 149063));
                b.get("draggable") && (_.N(a, 149069), b.get("clickable") === !1 && _.N(a, 149070));
                b.get("visible") === !1 && _.N(a, 149081);
                b.get("crossOnDrag") && _.N(a, 149067);
                b.get("cursor") && _.N(a, 149068);
                b.get("label") && _.N(a, 149080);
                b.get("title") && _.N(a, 149090);
                b.get("opacity") !=
                    null && _.N(a, 149082);
                b.get("optimized") === !0 ? _.N(a, 149085) : b.get("optimized") === !1 && _.N(a, 149071);
                b.get("zIndex") != null && _.N(a, 149092);
                c = b.get("icon");
                var d = new HR;
                (d = !c || c === d.icon.url || c.url === d.icon.url) ? _.N(a, 173084): _.N(a, 173083);
                typeof c === "string" ? _.N(a, 149079) : c && c.url != null ? (c.anchor && _.N(a, 149074), c.labelOrigin && _.N(a, 149075), c.origin && _.N(a, 149076), c.scaledSize && _.N(a, 149077), c.size && _.N(a, 149078)) : c && c.path != null ? (c = c.path, c === 0 ? _.N(a, 149088) : c === 1 ? _.N(a, 149072) : c === 2 ? _.N(a, 149073) : c ===
                    3 ? _.N(a, 149086) : c === 4 ? _.N(a, 149087) : _.N(a, 149089)) : CR(c) && _.N(a, 149083);
                b.get("shape") && (_.N(a, 149084), d && _.N(a, 162762));
                if (c = b.get("place")) c.placeId ? _.N(a, 149093) : _.N(a, 149094), b.get("attribution") && _.N(a, 149061)
            }
        },
        TVa = function(a, b, c) {
            SVa(a, c, d => {
                a.set(b, d);
                const e = d ? DR(d) : null;
                b === "viewIcon" && d && e && a.mh && a.mh(e, d.anchor, d.labelOrigin);
                d = a.get("modelLabel");
                a.set("viewLabel", d ? {
                    text: d.text || d,
                    color: _.ym(d.color, "#000000"),
                    fontWeight: _.ym(d.fontWeight, ""),
                    fontSize: _.ym(d.fontSize, "14px"),
                    fontFamily: _.ym(d.fontFamily,
                        "Roboto,Arial,sans-serif"),
                    className: d.className || ""
                } : null)
            })
        },
        SVa = function(a, b, c) {
            b ? CR(b) ? c(b) : b.path != null ? c(a.nh(b)) : (_.Am(b) || (b.size = b.size || b.scaledSize), b.size ? c(b) : (b.url || (b = {
                url: b
            }), IVa(b.url, d => {
                b.size = d || new _.No(24, 24);
                c(b)
            }))) : c(null)
        },
        UVa = function(a) {
            const b = a.get("mapPixelBoundsQ");
            var c = a.get("icon");
            const d = a.get("position");
            if (!b || !c || !d) return a.get("visible") !== !1;
            const e = c.anchor || _.cp,
                f = c.size.width + Math.abs(e.x);
            c = c.size.height + Math.abs(e.y);
            return d.x > b.minX - f && d.y > b.minY -
                c && d.x < b.maxX + f && d.y < b.maxY + c ? a.get("visible") !== !1 : !1
        },
        VVa = function(a, b) {
            a.origin = b;
            a.nh.ri()
        },
        IR = function(a) {
            a.mh && (_.Ax(a.mh), a.mh = null)
        },
        WVa = function(a, b, c) {
            _.XI(() => {
                a.style.webkitAnimationDuration = c.duration ? c.duration + "ms" : "";
                a.style.webkitAnimationIterationCount = `${c.nn}`;
                a.style.webkitAnimationName = b || ""
            })
        },
        XVa = function() {
            const a = [];
            for (let b = 0; b < JR.length; b++) {
                const c = JR[b];
                c.tick();
                c.mh || a.push(c)
            }
            JR = a;
            JR.length === 0 && (window.clearInterval(KR), KR = null)
        },
        LR = function(a) {
            return a ? a.__gm_at ||
                _.cp : null
        },
        ZVa = function(a, b) {
            var c = 1,
                d = a.animation;
            var e = d.frames[OVa(d, b)];
            var f;
            d = a.animation;
            (f = d.frames[OVa(d, b) + 1]) && (c = (b - e.time) / (f.time - e.time));
            b = LR(a.element);
            d = a.element;
            f ? (c = (0, YVa[e.sn || "linear"])(c), e = e.translate, f = f.translate, c = new _.Jo(Math.round(c * f[0] - c * e[0] + e[0]), Math.round(c * f[1] - c * e[1] + e[1]))) : c = new _.Jo(e.translate[0], e.translate[1]);
            c = d.__gm_at = c;
            d = c.x - b.x;
            b = c.y - b.y;
            if (d !== 0 || b !== 0) c = a.element, e = new _.Jo(_.xm(c.style.left) || 0, _.xm(c.style.top) || 0), e.x += d, e.y += b, _.sx(c, e);
            _.Wn(a,
                "tick")
        },
        bWa = function(a, b, c) {
            let d;
            var e;
            if (e = c.BH !== !1) e = _.Dq(), e = e.mh.uh || e.mh.rh && _.lw(e.mh.version, 7);
            e ? d = new $Va(a, b, c) : d = new aWa(a, b, c);
            d.start();
            return d
        },
        OR = function(a) {
            a.qh && (MR(a.Vh), a.qh.release(), a.qh = null);
            a.mh && _.Ax(a.mh);
            a.mh = null;
            a.ph && _.Ax(a.ph);
            a.ph = null;
            NR(a, !0);
            a.elements = []
        },
        PR = function(a) {
            return a.get("clickable") !== !1
        },
        NR = function(a, b = !1) {
            a.wh ? a.Ah = !0 : (_.Wn(a, b ? "ELEMENTS_REMOVED" : "CLEAR_TARGET"), a.targetElement && _.Ax(a.targetElement), a.targetElement = null, a.sh && (a.sh.unbindAll(),
                a.sh.release(), a.sh = null, MR(a.xh), a.xh = null), a.xj && a.xj.remove(), a.th && a.th.remove())
        },
        dWa = function(a, b) {
            const c = a.getIcon();
            if (c) {
                var d = c.url != null;
                a.mh && a.Nh === d && (_.Ax(a.mh), a.mh = null);
                a.Nh = !d;
                var e;
                d && (e = {
                    st: () => {}
                });
                a.mh = QR(a, b, a.mh, c, e);
                cWa(a, c, RR(a))
            }
        },
        hWa = function(a) {
            var b = a.getLabel();
            if (b) {
                if (!a.qh) {
                    const e = a.qh = new eWa(a.getPanes(), b, a.get("opacity"), a.get("visible"), a.si);
                    a.Vh = [_.Hn(a, "label_changed", () => {
                            e.setLabel(a.get("label"))
                        }), _.Hn(a, "opacity_changed", () => {
                            e.setOpacity(a.get("opacity"))
                        }),
                        _.Hn(a, "panes_changed", () => {
                            var f = a.get("panes");
                            e.xm = f;
                            IR(e);
                            e.nh.ri()
                        }), _.Hn(a, "visible_changed", () => {
                            e.setVisible(a.get("visible"))
                        })
                    ]
                }
                if (b = a.getIcon()) {
                    var c = a.mh,
                        d = RR(a);
                    c = fWa(a, b, d, LR(c) || _.cp);
                    d = DR(b);
                    d = b.labelOrigin || new _.Jo(d.width / 2, d.height / 2);
                    CR(b) && (b = b.getSize().width, d = new _.Jo(b / 2, b / 2));
                    VVa(a.qh, new _.Jo(c.x + d.x, c.y + d.y));
                    a.qh.setZIndex(gWa(a));
                    _.pq(a.qh.nh)
                }
            }
        },
        jWa = function(a) {
            if (!a.zh) {
                a.oh && (a.uh && _.Jn(a.uh), a.oh.cancel(), a.oh = null);
                var b = a.get("animation");
                if (b = iWa[b]) {
                    var c = b.options;
                    a.mh && (a.zh = !0, a.set("animating", !0), b = bWa(a.mh, b.icon, c), a.oh = b, a.uh = _.Sn(b, "done", () => {
                        a.set("animating", !1);
                        a.oh = null;
                        a.set("animation", null)
                    }))
                }
            }
        },
        MR = function(a) {
            if (a)
                for (let b = 0, c = a.length; b < c; b++) _.Jn(a[b])
        },
        RR = function(a) {
            return _.Dq().transform ? Math.min(1, a.get("scale") || 1) : 1
        },
        fWa = function(a, b, c, d) {
            const e = a.getPosition(),
                f = DR(b);
            var g = (b = SR(b)) ? b.x : f.width / 2;
            a.Ch.x = e.x + d.x - Math.round(g - (g - f.width / 2) * (1 - c));
            b = b ? b.y : f.height;
            a.Ch.y = e.y + d.y - Math.round(b - (b - f.height / 2) * (1 - c));
            return a.Ch
        },
        gWa =
        function(a) {
            let b = a.get("zIndex");
            a.Nl && (b = 1E6);
            _.vm(b) || (b = Math.min(a.getPosition().y, 999999));
            return b
        },
        SR = function(a) {
            return CR(a) ? a.getAnchor() : a.anchor
        },
        cWa = function(a, b, c) {
            const d = DR(b);
            a.size.width = c * d.width;
            a.size.height = c * d.height;
            a.set("size", a.size);
            const e = a.get("anchorPoint");
            if (!e || e.mh) b = SR(b), a.anchorPoint.x = c * (b ? d.width / 2 - b.x : 0), a.anchorPoint.y = -c * (b ? b.y : d.height), a.anchorPoint.mh = !0, a.set("anchorPoint", a.anchorPoint)
        },
        QR = function(a, b, c, d, e) {
            if (CR(d)) a = kWa(a, b, c, d);
            else if (d.url !=
                null) {
                const f = d.origin || _.cp;
                a = a.get("opacity");
                const g = _.ym(a, 1);
                c ? (c.firstChild.__src__ !== d.url && _.wL(c.firstChild, d.url), _.yL(c, d.size, f, d.scaledSize), c.firstChild.style.opacity = `${g}`) : (e = e || {}, e.UA = !_.Bq.th, e.alpha = !0, e.opacity = a, c = _.xL(d.url, null, f, d.size, null, d.scaledSize, e), c.style.display = "none", b.appendChild(c));
                a = c
            } else b = c || _.tx("div", b), b.textContent = "", c = _.Zr(), e = _.ox(b).createElement("canvas"), e.width = d.size.width * c, e.height = d.size.height * c, e.style.width = _.Em(d.size.width), e.style.height =
                _.Em(d.size.height), _.Eq(b, d.size), b.appendChild(e), _.sx(e, _.cp), _.Hq(e), e = e.getContext("2d"), e.lineCap = e.lineJoin = "round", e.scale(c, c), c = new _.zHa(e), e.beginPath(), c.ni(d.CG, d.anchor.x, d.anchor.y, d.rotation || 0, d.scale), d.fillOpacity && (e.fillStyle = d.fillColor, e.globalAlpha = d.fillOpacity, e.fill()), d.strokeWeight && (e.lineWidth = d.strokeWeight, e.strokeStyle = d.strokeColor, e.globalAlpha = d.strokeOpacity, e.stroke()), _.eJ(b, _.ym(a.get("opacity"), 1)), a = b;
            c = a;
            c.Ky = d;
            return c
        },
        lWa = function(a, b) {
            a.xj && a.th && a.element ===
                b || (a.element = b, a.xj && a.xj.remove(), a.th && a.th.remove(), a.xj = _.jy(b, {
                    zl(c) {
                        a.wh++;
                        _.Vx(c);
                        _.Wn(a, "mousedown", c.mh)
                    },
                    Ol(c) {
                        a.wh--;
                        !a.wh && a.Ah && _.WI(this, () => {
                            a.Ah = !1;
                            NR(a);
                            _.pq(a.rh)
                        }, 0);
                        _.Xx(c);
                        _.Wn(a, "mouseup", c.mh)
                    },
                    nl: ({
                        event: c,
                        Fq: d
                    }) => {
                        if (a.get("clickable") !== !1 || a.getDraggable()) _.Zw(c.mh), c.button === 3 ? d || c.button === 3 && _.Wn(a, "rightclick", c.mh) : d ? _.Wn(a, "dblclick", c.mh) : (_.Wn(a, "click", c.mh), _.N(window, 171150))
                    },
                    zu: c => {
                        _.Yx(c);
                        _.Wn(a, "contextmenu", c.mh)
                    }
                }), a.th = new _.LB(b, b, {
                    tt(c) {
                        _.Wn(a, "mouseout",
                            c)
                    },
                    ut(c) {
                        _.Wn(a, "mouseover", c)
                    }
                }))
        },
        kWa = function(a, b, c, d) {
            c = c || _.tx("div", b);
            _.Xq(c);
            b === a.getPanes().overlayMouseTarget ? (b = d.element.cloneNode(!0), _.eJ(b, 0), c.appendChild(b)) : c.appendChild(d.element);
            b = d.getSize();
            c.style.width = _.Lo(b);
            c.style.height = _.Mo(b);
            c.style.pointerEvents = "none";
            c.style.userSelect = "none";
            _.Sn(d, "changed", () => {
                a.nh()
            });
            return c
        },
        TR = function(a) {
            const b = a.marker.get("place");
            a = a.marker.get("position");
            return b && b.location || a
        },
        UR = function(a, b) {
            a.ph && a.ph.has(b) && ({
                    marker: a
                } =
                a.ph.get(b), b.zn = mWa(a), b.zn && (b = a.getMap()) && _.N(b, 184438))
        },
        oWa = function(a, b) {
            if (a.ph) {
                var {
                    iF: c,
                    marker: d
                } = a.ph.get(b);
                for (const e of nWa) c.push(FVa(d, e, () => {
                    UR(a, b)
                })), c.push(EVa(d, e, () => {
                    !mWa(d) && b.zn && UR(a, b)
                }))
            }
        },
        pWa = function(a) {
            const b = a.nh.__gm;
            a.mh.bindTo("mapPixelBounds", b, "pixelBounds");
            a.mh.bindTo("panningEnabled", a.nh, "draggable");
            a.mh.bindTo("panes", b)
        },
        qWa = function(a) {
            const b = a.nh.__gm;
            _.Hn(a.uh, "dragging_changed", () => {
                b.set("markerDragging", a.marker.get("dragging"))
            });
            b.set("markerDragging",
                b.get("markerDragging") || a.marker.get("dragging"))
        },
        sWa = function(a) {
            a.rh.push(_.Vn(a.mh, "panbynow", a.nh.__gm));
            rWa.forEach(b => {
                a.rh.push(_.Hn(a.mh, b, c => {
                    const d = a.wh ? TR(a) : a.marker.get("internalPosition");
                    c = new _.MB(d, c, a.mh.get("position"));
                    _.Wn(a.marker, b, c)
                }))
            })
        },
        tWa = function(a) {
            const b = () => {
                a.marker.get("place") ? a.mh.set("draggable", !1) : a.mh.set("draggable", !!a.marker.get("draggable"))
            };
            a.rh.push(_.Hn(a.uh, "draggable_changed", b));
            a.rh.push(_.Hn(a.uh, "place_changed", b));
            b()
        },
        uWa = function(a) {
            a.rh.push(_.Hn(a.nh,
                "projection_changed", () => {
                    VR(a)
                }));
            a.rh.push(_.Hn(a.uh, "position_changed", () => {
                VR(a)
            }));
            a.rh.push(_.Hn(a.uh, "place_changed", () => {
                VR(a)
            }))
        },
        wWa = function(a) {
            a.rh.push(_.Hn(a.mh, "dragging_changed", () => {
                if (a.mh.get("dragging")) a.zh = a.qh.ho(), a.zh && _.EM(a.qh, a.zh);
                else {
                    a.zh = null;
                    a.yh = null;
                    var b = a.qh.getPosition();
                    if (b && (b = _.Ar(b, a.nh.get("projection")), b = vWa(a, b))) {
                        const c = _.Cx(b, a.nh.get("projection"));
                        a.marker.get("place") || (a.xh = !1, a.marker.set("position", b), a.xh = !0);
                        a.qh.setPosition(c)
                    }
                }
            }));
            a.rh.push(_.Hn(a.mh,
                "deltaclientposition_changed", () => {
                    var b = a.mh.get("deltaClientPosition");
                    if (b && (a.zh || a.yh)) {
                        var c = a.yh || a.zh;
                        a.yh = {
                            clientX: c.clientX + b.clientX,
                            clientY: c.clientY + b.clientY
                        };
                        b = a.Mh.Nm(a.yh);
                        b = _.Ar(b, a.nh.get("projection"));
                        c = a.yh;
                        var d = vWa(a, b);
                        d && (a.marker.get("place") || (a.xh = !1, a.marker.set("position", d), a.xh = !0), d.equals(b) || (b = _.Cx(d, a.nh.get("projection")), c = a.qh.ho(b)));
                        c && _.EM(a.qh, c)
                    }
                }))
        },
        xWa = function(a) {
            if (a.oh) {
                a.mh.bindTo("scale", a.oh);
                a.mh.bindTo("position", a.oh, "pixelPosition");
                const b =
                    a.nh.__gm;
                a.oh.bindTo("latLngPosition", a.marker, "internalPosition");
                a.oh.bindTo("focus", a.nh, "position");
                a.oh.bindTo("zoom", b);
                a.oh.bindTo("offset", b);
                a.oh.bindTo("center", b, "projectionCenterQ");
                a.oh.bindTo("projection", a.nh)
            }
        },
        zWa = function(a) {
            if (a.oh) {
                const b = new yWa(a.nh instanceof _.$o);
                b.bindTo("internalPosition", a.oh, "latLngPosition");
                b.bindTo("place", a.marker);
                b.bindTo("position", a.marker);
                b.bindTo("draggable", a.marker);
                a.mh.bindTo("draggable", b, "actuallyDraggable")
            }
        },
        VR = function(a) {
            if (a.xh) {
                var b =
                    TR(a);
                b && a.qh.setPosition(_.Cx(b, a.nh.get("projection")))
            }
        },
        vWa = function(a, b) {
            const c = a.nh.__gm.get("snappingCallback");
            return c && (a = c({
                latLng: b,
                overlay: a.marker
            })) ? a : b
        },
        mWa = function(a) {
            return nWa.some(b => DVa(a, b))
        },
        BWa = function(a, b, c) {
            if (b instanceof _.po) {
                const d = b.__gm;
                Promise.all([d.nh, d.oh]).then(([{
                    Mh: e
                }, f]) => {
                    AWa(a, b, c, e, f)
                })
            } else AWa(a, b, c, null)
        },
        AWa = function(a, b, c, d, e = !1) {
            const f = new Map,
                g = h => {
                    var k = b instanceof _.po;
                    const n = k ? h.__gm.Sr.map : h.__gm.Sr.streetView,
                        p = n && n.nh === b,
                        r = p !== a.contains(h);
                    n && r && (k ? (h.__gm.Sr.map.dispose(), h.__gm.Sr.map = null) : (h.__gm.Sr.streetView.dispose(), h.__gm.Sr.streetView = null));
                    !a.contains(h) || !k && h.get("mapOnly") || p || (b instanceof _.po ? (k = b.__gm, h.__gm.Sr.map = new CWa(h, b, c, _.vM(k, h), d, k.Bh, f)) : h.__gm.Sr.streetView = new CWa(h, b, c, _.Lk, null, null, null), RVa(b, h, e))
                };
            _.Hn(a, "insert", g);
            _.Hn(a, "remove", g);
            a.forEach(g)
        },
        WR = function(a) {
            return _.Zr() / (a.webkitBackingStorePixelRatio || a.mozBackingStorePixelRatio || a.msBackingStorePixelRatio || a.oBackingStorePixelRatio || a.backingStorePixelRatio ||
                1)
        },
        DWa = function(a, b, c) {
            a = a.mh;
            a.width = b;
            a.height = c;
            return a
        },
        EWa = function(a) {
            const b = [];
            a.zj.forEach(c => {
                b.push(c)
            });
            b.sort((c, d) => c.zIndex - d.zIndex);
            return b
        },
        FWa = function(a) {
            const b = EWa(a),
                c = a.getContext(),
                d = WR(c);
            a = a.fi.size;
            c.clearRect(0, 0, Math.ceil(a.Sh * d), Math.ceil(a.Th * d));
            b.forEach(e => {
                c.globalAlpha = _.ym(e.opacity, 1);
                c.drawImage(e.image, e.Dz, e.Ez, e.Cz, e.wz, Math.round(e.dx * d), Math.round(e.dy * d), e.xr * d, e.ur * d)
            })
        },
        GWa = function(a, b, c) {
            if (c.dx > a || c.dy > b || c.dx + c.xr < a || c.dy + c.ur < b) a = !1;
            else a: {
                var d =
                    c.qw.shape;a -= c.dx;b -= c.dy;
                if (!d) throw Error("Shape cannot be null.");c = d.coords || [];
                switch (d.type.toLowerCase()) {
                    case "rect":
                        a = c[0] <= a && a <= c[2] && c[1] <= b && b <= c[3];
                        break a;
                    case "circle":
                        d = c[2];
                        a -= c[0];
                        b -= c[1];
                        a = a * a + b * b <= d * d;
                        break a;
                    default:
                        d = c, c = d.length, d[0] == d[c - 2] && d[1] == d[c - 1] || d.push(d[0], d[1]), a = _.cGa(a, b, d) != 0
                }
            }
            return a
        },
        YR = function(a, b, c, d) {
            var e = b.hj,
                f = a.nh.get();
            if (!f) return null;
            f = f.fi.size;
            c = _.FM(a.oh, e, new _.Jo(c, d));
            if (!c) return null;
            a = new _.Jo(c.ju.Zh * f.Sh, c.ju.ai * f.Th);
            const g = [];
            c.fl.zj.forEach(h =>
                g.push(h));
            g.sort((h, k) => k.zIndex - h.zIndex);
            c = null;
            for (e = 0; d = g[e]; ++e) {
                f = d.qw;
                if (f.clickable === !1) continue;
                f = f.KA;
                if (GWa(a.x, a.y, d)) {
                    c = f;
                    break
                }
            }
            c && (b.jk = d);
            return c
        },
        IWa = function(a, b) {
            if (!b.GB) {
                b.GB = !0;
                var c = _.zr(a.get("projection")),
                    d = b.su;
                if (d) {
                    d.dx < -64 || d.dy < -64 || d.dx + d.xr > 64 || d.dy + d.ur > 64 ? (_.rq(a.oh, b), d = a.nh.search(_.rt)) : (d = b.latLng, d = new _.Jo(d.lat(), d.lng()), b.hj = d, _.yM(a.ph, {
                        hj: d,
                        marker: b
                    }), d = _.ZFa(a.nh, d));
                    for (let f = 0, g = d.length; f < g; ++f) {
                        var e = d[f];
                        const h = e.fl || null;
                        if (e = HWa(a, h, e.tH ||
                                null, b, c)) b.zj[_.Zn(e)] = e, _.rq(h.zj, e)
                    }
                }
            }
        },
        JWa = function(a, b) {
            b.GB && (b.GB = !1, a.oh.contains(b) ? a.oh.remove(b) : a.ph.remove({
                hj: b.hj,
                marker: b
            }), _.qm(b.zj, (c, d) => {
                delete b.zj[c];
                d.fl.zj.remove(d)
            }))
        },
        KWa = function(a, b) {
            a.tiles[_.Zn(b)] = b;
            var c = {
                Zh: b.ej.x,
                ai: b.ej.y,
                ii: b.zoom
            };
            const d = _.zr(a.get("projection"));
            var e = _.Oy(a.fi, c);
            e = new _.Jo(e.mh, e.nh);
            const {
                min: f,
                max: g
            } = _.BI(a.fi, c, 64 / a.fi.size.Sh);
            c = _.sp(f.mh, f.nh, g.mh, g.nh);
            _.bGa(c, d, e, (h, k) => {
                h.tH = k;
                h.fl = b;
                b.bq[_.Zn(h)] = h;
                _.wM(a.nh, h);
                const n = (a.ph.search(h) || []).map(p => p.marker);
                a.oh.forEach(p => {
                    n.push(p)
                });
                for (let p = 0, r = n.length; p < r; ++p) {
                    k = n[p];
                    const t = HWa(a, b, h.tH, k, d);
                    t && (k.zj[_.Zn(t)] = t, _.rq(b.zj, t))
                }
            });
            b.div && b.zj && a.qh(b.div, b.zj)
        },
        LWa = function(a, b) {
            b && (delete a.tiles[_.Zn(b)], b.zj.forEach(c => {
                b.zj.remove(c);
                delete c.qw.zj[_.Zn(c)]
            }), _.qm(b.bq, (c, d) => {
                a.nh.remove(d)
            }))
        },
        HWa = function(a, b, c, d, e) {
            if (!(e && c && d.latLng && b)) return null;
            var f = e.fromLatLngToPoint(c);
            c = e.fromLatLngToPoint(d.latLng);
            if (!c || !f) return null;
            e = a.fi.size;
            a = _.FBa(a.fi, new _.Oq(c.x,
                c.y), new _.Oq(f.x, f.y), b.zoom);
            c.x = a.Zh * e.Sh;
            c.y = a.ai * e.Th;
            a = d.zIndex;
            _.vm(a) || (a = c.y);
            a = Math.round(a * 1E3) + Number(_.Zn(d)) % 1E3;
            f = d.su;
            b = {
                image: f.image,
                Dz: f.Dz,
                Ez: f.Ez,
                Cz: f.Cz,
                wz: f.wz,
                dx: f.dx + c.x,
                dy: f.dy + c.y,
                xr: f.xr,
                ur: f.ur,
                zIndex: a,
                opacity: d.opacity,
                fl: b,
                qw: d
            };
            return b.dx > e.Sh || b.dy > e.Th || b.dx + b.xr < 0 || b.dy + b.ur < 0 ? null : b
        },
        MWa = function(a) {
            return typeof a === "string" ? (ZR.has(a) || ZR.set(a, {
                url: a
            }), ZR.get(a)) : a
        },
        SWa = function(a, b, c) {
            const d = new _.jr,
                e = new _.jr,
                f = new NWa;
            new OWa(a, d, new HR, f, c);
            const g = _.ox(b.getDiv()).createElement("canvas"),
                h = {};
            a = _.sp(-100, -300, 100, 300);
            const k = new _.iN(a);
            a = _.sp(-90, -180, 90, 180);
            const n = _.aGa(a, (w, y) => w.marker === y.marker);
            let p = null,
                r = null;
            const t = new _.Yo(null),
                v = b.__gm;
            v.nh.then(w => {
                v.sh.register(new PWa(h, v, t, w.Mh.wk));
                _.jw(w.xs, y => {
                    if (y && p !== y.fi) {
                        r && r.unbindAll();
                        var E = p = y.fi;
                        r = new QWa(h, d, e, function(H, M) {
                            return new RWa(M, new $R(H, M, g, E), H)
                        }, k, n, p);
                        r.bindTo("projection", b);
                        t.set(r.mh())
                    }
                })
            });
            _.GM(b, t, "markerLayer", -1)
        },
        UWa = function(a) {
            a.Qt || (a.Qt = _.XI(() => {
                a.Qt = 0;
                const b = a.Ev;
                a.Ev = {};
                const c =
                    a.Lw;
                for (const d of Object.values(b)) TWa(a, d);
                c && !a.Lw && a.hu.forEach(d => {
                    TWa(a, d)
                })
            }))
        },
        TWa = function(a, b) {
            var c = b.get("place");
            c = c ? c.location : b.get("position");
            b.set("internalPosition", c);
            b.changed = a.nM;
            if (!b.get("animating"))
                if (a.eD.remove(b), !c || b.get("visible") == 0 || b.__gm && b.__gm.Wo) a.hu.remove(b);
                else {
                    a.Lw && !a.jF && a.hu.getSize() >= 256 && (a.Lw = !1);
                    c = b.get("optimized");
                    const e = b.get("draggable"),
                        f = !!b.get("animation");
                    var d = b.get("icon");
                    const g = !!d && d.path != null;
                    d = CR(d);
                    const h = b.get("label") != null;
                    a.jF || c == 0 || e || f || g || d || h || !c && a.Lw ? _.rq(a.hu, b) : (a.hu.remove(b), _.rq(a.eD, b))
                }
        },
        VWa = function(a, b) {
            const c = new _.eq;
            c.onAdd = () => {};
            c.onContextLost = () => {};
            c.onRemove = () => {};
            c.onContextRestored = () => {};
            c.onDraw = ({
                transformer: d
            }) => {
                a.onDraw(d)
            };
            _.Ut.add(c);
            c.setMap(b);
            return c
        },
        WWa = function(a) {
            a.sh || (a.sh = setTimeout(() => {
                const b = [...a.ph].filter(c => !c.Pp).length;
                b > 0 && a.ki.Eh(a.map, b);
                a.sh = 0
            }, 0))
        },
        ZWa = function(a, b) {
            a.qh.has(b) || (a.qh.add(b), _.rz(_.qz(), () => {
                if (a.map) {
                    var c = [];
                    for (const d of a.qh) {
                        if (!d.map) continue;
                        const e = d.targetElement;
                        e.parentNode || c.push(d);
                        const f = d.Wo !== !1 && aS(d) || d.Zs,
                            g = _.qo(a.map);
                        g || (a.th || (a.th = a.nh.attachShadow({
                            mode: _.zq[166] ? "open" : "closed"
                        }), a.th.append(a.rh, a.mh)), a.nh.append(e));
                        g && e.parentElement === g || !g && e.parentElement === a.nh ? e.setAttribute("slot", f ? XWa : YWa) : e.style.visibility = f ? "hidden" : "";
                        d.Ym(!f);
                        d.uw = !1
                    }
                    a.qh.clear();
                    for (const d of c) d.Kz(!0)
                }
            }))
        },
        $Wa = function(a) {
            bS || (bS = new ResizeObserver(b => {
                for (const c of b) c.target.dispatchEvent(new CustomEvent("marker-resize", {
                    detail: c.contentRect
                }))
            }));
            bS.observe(a)
        },
        cXa = function(a, b) {
            const c = _.Ka(b);
            let d = cS.get(c);
            d || (d = new aXa(b), cS.set(c, d));
            b = d;
            bXa(a, b.Xn);
            b.ph.add(a);
            WWa(b);
            $Wa(a.targetElement)
        },
        dXa = function(a) {
            a = _.Ka(a);
            (a = cS.get(a)) && a.requestRedraw()
        },
        eXa = function(a, b) {
            b = _.Ka(b);
            (b = cS.get(b)) && ZWa(b, a)
        },
        fXa = function(a) {
            let b = 0,
                c = 0;
            for (const d of a) switch (d) {
                case "ArrowLeft":
                    --b;
                    break;
                case "ArrowRight":
                    b += 1;
                    break;
                case "ArrowDown":
                    c += 1;
                    break;
                case "ArrowUp":
                    --c
            }
            return {
                deltaX: b,
                deltaY: c
            }
        },
        eS = function(a, b, c = !0) {
            a.mh.position = a.yh;
            dS(a, b, c)
        },
        dS = function(a, b, c = !0) {
            b.preventDefault();
            b.stopImmediatePropagation();
            fS(a);
            gXa(a);
            a.nh && (a.nh.release(), a.nh = null);
            c && gS(a.mh, "dragend", b)
        },
        hS = function(a, b) {
            if (b) {
                a.oh.style.display = "";
                b = a.mh.anchorTop || "-100%";
                let c = a.mh.anchorLeft || "-50%";
                b === "0" && (b = "0px");
                c === "0" && (c = "0px");
                a.oh.style.left = `calc(-1 * ${c})`;
                a.oh.style.top = `calc(-1 * ${b})`
            } else a.oh.style.display = "none"
        },
        iXa = function(a) {
            a.oh.style.opacity = "0.5";
            a.oh.style.position = "absolute";
            hS(a, !1);
            a.oh.style.transform = "translate(-50%, -50%)";
            a.oh.style.zIndex = "-1";
            hXa(a);
            const b = a.mh.Ep;
            b.addEventListener("pointerenter", a.Ch);
            b.addEventListener("pointerleave", a.Fh);
            b.addEventListener("focus", a.Ch);
            b.addEventListener("blur", a.Fh)
        },
        jXa = function(a, b = !1) {
            return a.ph ? _.Bz : b ? "pointer" : _.uma
        },
        iS = function(a) {
            const b = a.mh.vk;
            b && b.appendChild(a.oh)
        },
        hXa = function(a) {
            a.oh.children[0] ? .remove();
            var b = a.mh,
                c;
            if (!(c = b.dragIndicator)) {
                if (!b.Bv) {
                    const {
                        url: d,
                        scaledSize: e
                    } = (new HR).mh;
                    b.Bv = new Image(e.width, e.height);
                    b.Bv.src = d;
                    b.Bv.alt = ""
                }
                c = b.Bv
            }
            a.oh.appendChild(c);
            iS(a)
        },
        lXa = function(a) {
            if (!a.mh.DA) {
                a.nh = new _.jN((c, d) => {
                    var e = a.mh;
                    e.ui && _.Wn(e.ui, "panbynow", c, d)
                });
                _.DM(a.nh, !0);
                var b = kXa(a.mh);
                _.CM(a.nh, b);
                a.nh.ph = a.qh
            }
        },
        mXa = function(a, b) {
            fS(a);
            a.qh = !1;
            a.nh && (a.nh.ph = !1);
            a.rh = a.mh.ho();
            a.wh = _.ZL(b)
        },
        nXa = function(a, b) {
            var c = _.ZL(b);
            if (c) {
                b = c.clientX;
                c = c.clientY;
                var d = b - a.wh.clientX,
                    e = c - a.wh.clientY;
                a.wh = {
                    clientX: b,
                    clientY: c
                };
                b = {
                    clientX: a.rh.clientX + d,
                    clientY: a.rh.clientY + e
                };
                a.rh = b;
                a.mh.TC(b)
            }
        },
        oXa = function(a, b) {
            a.rh = a.mh.ho();
            a.yh = a.mh.position;
            a.wh = _.ZL(b);
            a.ph = !0;
            lXa(a);
            a.mh.Ep.setAttribute("aria-grabbed", "true");
            jS(a.mh);
            a.mh.Ep.style.zIndex = "2147483647";
            a.oh.style.opacity = "1";
            hS(a, !0);
            gS(a.mh, "dragstart", b)
        },
        pXa = function(a) {
            a.qh && (a.rh = a.mh.ho())
        },
        kS = function(a) {
            _.iy !== 2 ? (document.removeEventListener("pointermove", a.Ah), document.removeEventListener("pointerup", a.sh), document.removeEventListener("pointercancel", a.sh)) : (document.removeEventListener("touchmove", a.Ah, {
                passive: !1
            }), document.removeEventListener("touchend", a.sh), document.removeEventListener("touchcancel",
                a.sh), document.removeEventListener("touchstart", a.sh));
            fS(a);
            gXa(a);
            a.nh && (a.nh.release(), a.nh = null)
        },
        fS = function(a) {
            const b = a.mh.Ep;
            b.removeEventListener("keydown", a.Vh);
            b.removeEventListener("keyup", a.Xh);
            b.removeEventListener("blur", a.Oh)
        },
        qXa = function(a) {
            if (a.zh.size === 0) a.xh = 0;
            else {
                var {
                    deltaX: b,
                    deltaY: c
                } = fXa(a.zh), d = 1;
                _.hM(a.Hh) && (d = a.Hh.next());
                var e = Math.round(3 * d * b);
                d = Math.round(3 * d * c);
                e === 0 && (e = b);
                d === 0 && (d = c);
                e = {
                    clientX: a.rh.clientX + e,
                    clientY: a.rh.clientY + d
                };
                a.rh = e;
                a.mh.TC(e);
                a.xh = window.setTimeout(() => {
                    qXa(a)
                }, 10)
            }
        },
        gXa = function(a) {
            a.ph = !1;
            a.qh = !1;
            a.wh = null;
            a.rh = null;
            clearTimeout(a.xh);
            a.xh = 0;
            a.yh = null;
            a.Eh = null;
            a.uh = null;
            const b = a.mh.Ep,
                c = a.mh.zIndex;
            a.oh.style.opacity = "0.5";
            b.setAttribute("aria-grabbed", "false");
            b.style.zIndex = c == null ? "" : `${c}`;
            rXa(a.mh)
        },
        bXa = function(a, b) {
            a.GA = b;
            if (a.av) {
                var c = a.getAttribute("aria-describedby");
                c = c ? c.split(" ") : [];
                c.push(b);
                a.setAttribute("aria-describedby", c.join(" "))
            }
        },
        aS = function(a) {
            return a.collisionBehavior !== "REQUIRED" && !a.Nl && !!a.map && !!a.position
        },
        kXa = function(a) {
            return a.ui ? a.ui.get("pixelBounds") : null
        },
        gS = function(a, b, c) {
            _.Wn(a, b, new _.MB(a.Sp, c, a.lw ? new _.Jo(a.lw.Sh, a.lw.Th) : null))
        },
        jS = function(a) {
            _.Wn(a, "REMOVE_COLLISION")
        },
        rXa = function(a) {
            a.style.cursor = a.Hj ? jXa(a.Hj, a.cw) : a.cw ? "pointer" : ""
        },
        lS = function(a, b = !1) {
            aS(a) && (a.ui && BVa(a.ui.Hh, a), _.Wn(a, "UPDATE_MARKER_COLLISION"), b && a.zx && _.Wn(a, "UPDATE_BASEMAP_COLLISION"))
        },
        sXa = function(a) {
            a.cq.then(() => {
                _.Gr(a, "marker-view");
                a.style.position = "absolute";
                a.style.left = "0px"
            })
        },
        tXa = function(a) {
            if (a.kn.get("isOpen")) {
                var {
                    width: b
                } =
                a.getBoundingClientRect();
                a.kn.set("anchorPoint", new _.Jo(b ? b / 2 : 0, 0))
            }
        },
        mS = function(a) {
            a.style.pointerEvents = a.Qy ? "none" : a.MF ? "auto" : ""
        },
        nS = function(a) {
            a.zn = a.cw || !!a.av
        },
        uXa = function(a, b = !1) {
            if (a.Zs || !a.Rk) a.kn.set("pixelPosition", null);
            else if (a.kn.get("isOpen"))
                if (b) requestAnimationFrame(() => {
                    const {
                        offsetX: c,
                        offsetY: d
                    } = ER(a.Wx);
                    a.kn.set("pixelPosition", new _.Jo(c, d))
                });
                else {
                    const {
                        offsetX: c,
                        offsetY: d
                    } = ER(a.Wx);
                    a.kn.set("pixelPosition", new _.Jo(c, d))
                }
        },
        vXa = function(a, b) {
            var c;
            if (c = a.Hj) c = a.Hj,
                c = c.uh && b.timeStamp - c.uh >= 500 ? !0 : c.th;
            if (!c && a.Sp) {
                a.gmpDraggable || a.focus();
                gS(a, "click", b);
                if (a.gmpClickable || _.In(a, "gmp-click")) c = new oS, _.Wn(a, "gmp-click", c), a.dispatchEvent(c);
                a.ki.th(b)
            }
        },
        wXa = function(a) {
            !a.xj && a.map && a.ui && (a.YC = !0, a.xj = _.jy(a, {
                nl: ({
                    event: b,
                    Fq: c
                }) => {
                    a.MF ? (_.Zw(b.mh), b.button === 3 || c || vXa(a, b.mh)) : a === b.mh.target || a.Qy || (console.debug('To make AdvancedMarkerElement clickable and provide better accessible experiences, use addListener() to register a "click" event on the AdvancedMarkerElement instance.'),
                        a.ki.wh(a.map))
                }
            }), a.VA = _.Bma({
                draggable: a.xF,
                OE: new _.cC(a.map, "gestureHandling"),
                xl: a.ui.rm
            }), _.jw(a.VA, a.mF), a.YC = !1)
        },
        pS = function(a) {
            const b = c => c.nodeType === Node.TEXT_NODE && c.nodeValue != null && !/\S/.test(c.nodeValue);
            return a.childNodes.length > 0 ? ([...a.childNodes].every(b) && _.Bn(_.Wp(a, "AdvancedMarkerElement is displaying empty text content. If you want a pin to appear, make sure to remove any whitespace between the <gmp-advanced-marker> tags.")), [...a.childNodes]) : a.kl && a.kl.contains(a.pm) ? [a.pm] : []
        },
        xXa = function(a, b, c) {
            if (b && c && ({
                    altitude: b
                } = new _.Ep(b), b > 0 || b < 0)) throw a.ki.xh(window), _.Rm("Draggable AdvancedMarkerElement with non-zero altitude is not supported");
        },
        qS = function(a) {
            if (a.Mk) {
                const b = _.Ka(a.Mk),
                    c = cS.get(b);
                c && (c.ph.delete(a), c.isEmpty() && (c.dispose(), cS.delete(b)));
                bS && bS.unobserve(a.targetElement);
                _.Wn(a, "REMOVE_FOCUS");
                _.Wn(a, "REMOVE_COLLISION");
                a.Mh && (a.Kk && (a.Mh.Ql(a.Kk), a.Kk = null), a.Mh = null);
                a.Hj && kS(a.Hj);
                a.rE ? .remove();
                a.OH ? .remove();
                a.IG ? .remove();
                a.jG ? .remove();
                a.VA ? .removeListener(a.mF);
                a.xj && (a.xj.remove(), a.xj = null);
                a.kn.set("map", null);
                a.zx = null;
                a.ui = null;
                a.Mk = null;
                a.uw = !0
            }
        },
        rS = function(a) {
            if (a.ui && !a.Nl) {
                var b = a.ui.Bh;
                b && (a.zn && a.Rk && !a.Wo ? b.Hh(a) : _.Wn(a, "REMOVE_FOCUS"))
            }
        },
        zXa = function(a) {
            var b = a.ui.get("baseMapType");
            b = b && (!b.mapTypeId || !Object.values(_.Ls).includes(b.mapTypeId));
            a.zx = a.OF && !b;
            if (!a.Pp || a.position) a.zx ? dXa(a.map) : yXa(a)
        },
        AXa = function(a) {
            if (!a.Pp) {
                var b = a.ui.mh;
                b.UB.then(() => {
                    const c = _.aq(b, "ADVANCED_MARKERS");
                    if (!c.isAvailable) {
                        a.ui && a.ui.bi();
                        for (const d of c.mh) b.log(d);
                        a.ki.uh(a.map);
                        a.dispose()
                    }
                })
            }
        },
        BXa = function(a) {
            a.ki.Ch(a.map);
            a.ki.ph(a.map, a.Qy);
            if (a.cw) {
                const b = _.In(a, "gmp-click");
                a.ki.nh(a.map, b)
            }
            a.gmpDraggable && a.ki.qh(a.map);
            a.title && a.ki.rh(a.map);
            a.zIndex !== null && a.ki.sh(a.map);
            a.Il() > 0 && a.ki.mh(a.map);
            a.ki.oh(a.map, a.collisionBehavior)
        },
        sS = function(a, b) {
            a.Hj && pXa(a.Hj);
            if (b) {
                a.style.transform = `${`translate(${a.anchorLeft||"-50%"}, ${a.anchorTop||"-100%"})`} ${`translate(${b.x}px, ${b.y}px)`}`;
                const c = a.style.willChange ? a.style.willChange.replace(/\s+/g,
                    "").split(",") : [];
                c.includes("transform") || _.rz(_.qz(), () => {
                    c.push("transform");
                    a.style.willChange = c.join(",")
                }, a, a)
            }
            a.Rk = b;
            rS(a)
        },
        yXa = function(a) {
            var b = LVa(a.Mk, a.Sp);
            a.Kk ? a.Kk.setPosition(b, a.Il()) : a.Mh && (b = new _.lN(a.Mh.wk, a, b, a.Mh, null, a.Il(), a.IK), a.Mh.Cj(b), a.Kk = b)
        };
    _.Jo.prototype.Xy = _.da(15, function() {
        return Math.sqrt(this.x * this.x + this.y * this.y)
    });
    var nWa = ["click", "dblclick", "rightclick", "contextmenu"],
        CXa = class extends _.$n {
            constructor() {
                super();
                this.constraint = 0;
                this.mh = !1
            }
            position_changed() {
                this.mh || (this.mh = !0, this.set("rawPosition", this.get("position")), this.mh = !1)
            }
            rawPosition_changed() {
                if (!this.mh) {
                    this.mh = !0;
                    var a = this.set,
                        b;
                    var c = this.get("rawPosition");
                    if (c) {
                        (b = this.get("snappingCallback")) && (c = b(c));
                        b = c.x;
                        c = c.y;
                        var d = this.get("referencePosition");
                        d && (this.constraint === 2 ? b = d.x : this.constraint === 1 && (c = d.y));
                        b = new _.Jo(b, c)
                    } else b = null;
                    a.call(this, "position", b);
                    this.mh = !1
                }
            }
        },
        DXa = class {
            constructor(a, b, c, d, e = 0, f = 0) {
                this.width = c;
                this.height = d;
                this.offsetX = e;
                this.offsetY = f;
                this.nh = new Float64Array(2);
                this.nh[0] = a;
                this.nh[1] = b;
                this.mh = new Float32Array(2)
            }
            transform(a) {
                a.Tu(1, this.nh, this.mh, 0, 0, 0);
                this.mh[0] += this.offsetX;
                this.mh[1] += this.offsetY
            }
            isVisible(a) {
                return this.mh[0] >= -this.width && this.mh[0] <= a.width + this.width && this.mh[1] >= -this.height && this.mh[1] <= a.height + this.height
            }
            equals(a) {
                return this.nh[0] === a.nh[0] && this.nh[1] === a.nh[1] &&
                    this.width === a.width && this.height === a.height && this.offsetX === a.offsetX && this.offsetY === a.offsetY
            }
            oh(a) {
                return this.mh[0] > a.right || this.mh[0] + this.width < a.left || this.mh[1] > a.bottom || this.mh[1] + this.height < a.top ? !1 : !0
            }
        },
        YVa = {
            linear: a => a,
            ["ease-out"]: a => 1 - Math.pow(a - 1, 2),
            ["ease-in"]: a => Math.pow(a, 2)
        },
        tS = class {
            constructor(a) {
                this.frames = a;
                this.mh = ""
            }
        },
        GR;
    var iWa = {
        [1]: {
            options: {
                duration: 700,
                nn: "infinite"
            },
            icon: new tS([{
                time: 0,
                translate: [0, 0],
                sn: "ease-out"
            }, {
                time: .5,
                translate: [0, -20],
                sn: "ease-in"
            }, {
                time: 1,
                translate: [0, 0],
                sn: "ease-out"
            }])
        },
        [2]: {
            options: {
                duration: 500,
                nn: 1
            },
            icon: new tS([{
                time: 0,
                translate: [0, -500],
                sn: "ease-in"
            }, {
                time: .5,
                translate: [0, 0],
                sn: "ease-out"
            }, {
                time: .75,
                translate: [0, -20],
                sn: "ease-in"
            }, {
                time: 1,
                translate: [0, 0],
                sn: "ease-out"
            }])
        },
        [3]: {
            options: {
                duration: 200,
                Xy: 20,
                nn: 1,
                BH: !1
            },
            icon: new tS([{
                time: 0,
                translate: [0, 0],
                sn: "ease-in"
            }, {
                time: 1,
                translate: [0, -20],
                sn: "ease-out"
            }])
        },
        [4]: {
            options: {
                duration: 500,
                Xy: 20,
                nn: 1,
                BH: !1
            },
            icon: new tS([{
                time: 0,
                translate: [0, -20],
                sn: "ease-in"
            }, {
                time: .5,
                translate: [0, 0],
                sn: "ease-out"
            }, {
                time: .75,
                translate: [0, -10],
                sn: "ease-in"
            }, {
                time: 1,
                translate: [0, 0],
                sn: "ease-out"
            }])
        }
    };
    var HR = class {
        constructor() {
            this.icon = {
                url: _.$r("api-3/images/spotlight-poi3", !0),
                scaledSize: new _.No(26, 37),
                origin: new _.Jo(0, 0),
                anchor: new _.Jo(13, 37),
                labelOrigin: new _.Jo(13, 14)
            };
            this.nh = {
                url: _.$r("api-3/images/spotlight-poi-dotless3", !0),
                scaledSize: new _.No(26, 37),
                origin: new _.Jo(0, 0),
                anchor: new _.Jo(13, 37),
                labelOrigin: new _.Jo(13, 14)
            };
            this.mh = {
                url: _.$r("api-3/images/drag-cross", !0),
                scaledSize: new _.No(13, 11),
                origin: new _.Jo(0, 0),
                anchor: new _.Jo(7, 6)
            };
            this.shape = {
                coords: [13, 0, 4, 3.5, 0, 12, 2.75, 21,
                    13, 37, 23.5, 21, 26, 12, 22, 3.5
                ],
                type: "poly"
            }
        }
    };
    var EXa = class extends _.$n {
            constructor(a, b) {
                super();
                this.nh = a;
                this.mh = b;
                uS || (uS = new HR)
            }
            changed(a) {
                a !== "modelIcon" && a !== "modelShape" && a !== "modelCross" && a !== "modelLabel" || _.rz(_.qz(), this.oh, this, this)
            }
            oh() {
                const a = this.get("modelIcon");
                var b = this.get("modelLabel");
                TVa(this, "viewIcon", a || b && uS.nh || uS.icon);
                TVa(this, "viewCross", uS.mh);
                b = this.get("useDefaults");
                let c = this.get("modelShape");
                c || a && !b || (c = uS.shape);
                this.get("viewShape") !== c && this.set("viewShape", c)
            }
        },
        uS;
    var FXa = class extends _.$n {
        constructor() {
            super();
            this.nh = !1;
            this.mh = UVa(this);
            this.set("shouldRender", this.mh)
        }
        changed() {
            if (!this.nh) {
                var a = UVa(this);
                this.mh !== a && (this.mh = a, this.nh = !0, this.set("shouldRender", this.mh), this.nh = !1)
            }
        }
    };
    var yWa = class extends _.$n {
        constructor(a) {
            super();
            this.nh = a;
            this.mh = !1
        }
        internalPosition_changed() {
            if (!this.mh) {
                this.mh = !0;
                var a = this.get("position"),
                    b = this.get("internalPosition");
                a && b && !a.equals(b) && this.set("position", this.get("internalPosition"));
                this.mh = !1
            }
        }
        draggable_changed() {
            if (!this.mh) {
                this.mh = !0;
                if (this.nh) {
                    const a = this.get("place");
                    a ? this.set("internalPosition", a.location) : this.set("internalPosition", this.get("position"))
                }
                this.get("place") ? this.set("actuallyDraggable", !1) : this.set("actuallyDraggable",
                    this.get("draggable"));
                this.mh = !1
            }
        }
        position_changed() {
            this.draggable_changed()
        }
        place_changed() {
            this.draggable_changed()
        }
    };
    var eWa = class {
        constructor(a, b, c, d, e) {
            this.xm = a;
            this.label = b;
            this.opacity = c;
            this.visible = d;
            this.origin = void 0;
            this.zIndex = 0;
            this.ph = this.qh = this.mh = null;
            this.nh = new _.oq(this.rh, 0, this);
            this.oh = e;
            this.xm = a;
            this.label = b;
            this.opacity = c;
            this.visible = d
        }
        setOpacity(a) {
            this.opacity = a;
            this.nh.ri()
        }
        setLabel(a) {
            this.label = a;
            this.nh.ri()
        }
        setVisible(a) {
            this.visible = a;
            this.nh.ri()
        }
        setZIndex(a) {
            this.zIndex = a;
            this.nh.ri()
        }
        release() {
            this.xm = null;
            IR(this)
        }
        rh() {
            if (this.xm && this.label && this.visible !== !1) {
                var a = this.xm.markerLayer,
                    b = this.label;
                this.mh ? a.appendChild(this.mh) : (this.mh = document.createElement("div"), a.appendChild(this.mh), this.mh.style.transform = "translateZ(0)");
                a = this.mh;
                this.origin && _.sx(a, this.origin);
                var c = a.firstElementChild;
                c || (c = document.createElement("div"), a.appendChild(c), c.style.height = "100px", c.style.transform = "translate(-50%, -50px)", c.style.display = "table", c.style.borderSpacing = "0");
                let d = c.firstElementChild;
                d || (d = document.createElement("div"), c.appendChild(d), d.style.display = "table-cell", d.style.verticalAlign =
                    "middle", d.style.whiteSpace = "nowrap", d.style.textAlign = "center");
                c = d.firstElementChild || _.tx("div", d);
                c.textContent = b.text;
                c.style.color = b.color;
                c.style.fontSize = b.fontSize;
                c.style.fontWeight = b.fontWeight;
                c.style.fontFamily = b.fontFamily;
                c.className = b.className;
                c.setAttribute("aria-hidden", "true");
                if (this.oh && b !== this.ph) {
                    this.ph = b;
                    const {
                        width: e,
                        height: f
                    } = c.getBoundingClientRect();
                    b = new _.No(e, f);
                    b.equals(this.qh) || (this.qh = b, this.oh(b))
                }
                _.eJ(c, _.ym(this.opacity, 1));
                _.ux(a, this.zIndex)
            } else IR(this)
        }
    };
    var $Va = class {
        constructor(a, b, c) {
            this.element = a;
            this.animation = b;
            this.options = c;
            this.nh = !1;
            this.mh = null
        }
        start() {
            this.options.nn = this.options.nn || 1;
            this.options.duration = this.options.duration || 1;
            _.Qn(this.element, "webkitAnimationEnd", () => {
                this.nh = !0;
                _.Wn(this, "done")
            });
            WVa(this.element, PVa(this.animation), this.options)
        }
        cancel() {
            this.mh && (this.mh.remove(), this.mh = null);
            WVa(this.element, null, {});
            _.Wn(this, "done")
        }
        stop() {
            this.nh || (this.mh = _.Qn(this.element, "webkitAnimationIteration", () => {
                this.cancel()
            }))
        }
    };
    var JR = [],
        KR = null,
        aWa = class {
            constructor(a, b, c) {
                this.element = a;
                this.animation = b;
                this.nn = -1;
                this.mh = !1;
                this.startTime = 0;
                c.nn !== "infinity" && (this.nn = c.nn || 1);
                this.duration = c.duration || 1E3
            }
            start() {
                JR.push(this);
                KR || (KR = window.setInterval(XVa, 10));
                this.startTime = Date.now();
                this.tick()
            }
            cancel() {
                this.mh || (this.mh = !0, ZVa(this, 1), _.Wn(this, "done"))
            }
            stop() {
                this.mh || (this.nn = 1)
            }
            tick() {
                if (!this.mh) {
                    var a = Date.now();
                    ZVa(this, (a - this.startTime) / this.duration);
                    a >= this.startTime + this.duration && (this.startTime = Date.now(),
                        this.nn !== "infinite" && (this.nn--, this.nn || this.cancel()))
                }
            }
        };
    var GXa = _.za.DEF_DEBUG_MARKERS,
        vS = class extends _.$n {
            constructor(a, b, c) {
                super();
                this.qh = this.mh = null;
                this.Vh = [];
                this.Nh = !1;
                this.targetElement = null;
                this.Oh = !1;
                this.ph = null;
                this.elements = [];
                this.zh = !0;
                this.wh = 0;
                this.oh = this.Ph = this.bi = this.Xh = null;
                this.Ah = !1;
                this.element = this.th = this.xj = this.sh = this.uh = this.xh = null;
                this.Bh = !1;
                this.tq = null;
                this.rh = new _.oq(() => {
                    var d = this.get("panes"),
                        e = this.get("scale");
                    if (!d || !this.getPosition() || this.getVisible() === !1 || _.vm(e) && e < .1 && !this.Nl) OR(this);
                    else {
                        dWa(this,
                            d.markerLayer);
                        if (!this.wh) {
                            var f = this.getIcon();
                            if (f) {
                                var g = f.url;
                                e = PR(this);
                                var h = this.getDraggable(),
                                    k = this.get("title") || "",
                                    n = k;
                                n || (n = (n = this.getLabel()) ? n.text : "");
                                if (e || h || n) {
                                    var p = !e && !h && !k,
                                        r = CR(f),
                                        t = SR(f),
                                        v = this.get("shape"),
                                        w = DR(f),
                                        y = {};
                                    if (_.wx()) f = w.width, w = w.height, r = new _.No(f + 16, w + 16), f = {
                                        url: _.JB,
                                        size: r,
                                        anchor: t ? new _.Jo(t.x + 8, t.y + 8) : new _.Jo(Math.round(f / 2) + 8, w + 8),
                                        scaledSize: r
                                    };
                                    else {
                                        const E = f.scaledSize || w;
                                        (_.Bq.nh || _.Bq.mh) && v && (y.shape = v, w = E);
                                        if (!r || v) f = {
                                            url: _.JB,
                                            size: w,
                                            anchor: t,
                                            scaledSize: E
                                        }
                                    }
                                    t = f.url != null;
                                    this.Oh === t && NR(this);
                                    this.Oh = !t;
                                    y = this.targetElement = QR(this, this.getPanes().overlayMouseTarget, this.targetElement, f, y);
                                    this.targetElement.style.pointerEvents = p ? "none" : "";
                                    if (p = y.querySelector("img")) p.style.removeProperty("position"), p.style.removeProperty("opacity"), p.style.removeProperty("left"), p.style.removeProperty("top");
                                    t = y;
                                    p = null;
                                    (f = t.getAttribute("usemap") || t.firstChild && t.firstChild.getAttribute("usemap")) && f.length && (t = _.ox(t).getElementById(f.substring(1))) &&
                                        (p = t.firstChild);
                                    p && (p.tabIndex = -1, p.style.display = "inline", p.style.position = "absolute", p.style.left = "0px", p.style.top = "0px");
                                    GXa && (y.dataset.debugMarkerImage = g);
                                    y = p || y;
                                    y.title = k;
                                    n && this.xq().setAttribute("aria-label", n);
                                    this.Zw();
                                    h && !this.sh && (g = this.sh = new _.AHa(y, this.Eh, this.targetElement), this.Eh ? (g.bindTo("deltaClientPosition", this), g.bindTo("position", this)) : g.bindTo("position", this.yh, "rawPosition"), g.bindTo("containerPixelBounds", this, "mapPixelBounds"), g.bindTo("anchorPoint", this), g.bindTo("size",
                                        this), g.bindTo("panningEnabled", this), this.xh || (this.xh = [_.Vn(g, "dragstart", this), _.Vn(g, "drag", this), _.Vn(g, "dragend", this), _.Vn(g, "panbynow", this)]));
                                    g = this.get("cursor") || "pointer";
                                    h ? this.sh.set("draggableCursor", g) : y.style.cursor = e ? g : "";
                                    lWa(this, y)
                                }
                            }
                        }
                        d = d.overlayLayer;
                        if (h = e = this.get("cross")) h = this.get("crossOnDrag"), h === void 0 && (h = this.get("raiseOnDrag")), h = h !== !1 && this.getDraggable() && this.Nl;
                        h ? this.ph = QR(this, d, this.ph, e) : (this.ph && _.Ax(this.ph), this.ph = null);
                        this.elements = [this.mh, this.ph,
                            this.targetElement
                        ];
                        hWa(this);
                        for (e = 0; e < this.elements.length; ++e)
                            if (h = this.elements[e]) d = h, g = h.Ky, k = LR(h) || _.cp, h = RR(this), g = fWa(this, g, h, k), _.sx(d, g), (g = _.Dq().transform) && d.style.setProperty(g, h !== 1 ? `scale(${h}) ` : ""), d && _.ux(d, gWa(this));
                        jWa(this);
                        for (d = 0; d < this.elements.length; ++d)(e = this.elements[d]) && _.cJ(e);
                        _.Wn(this, "UPDATE_FOCUS")
                    }
                }, 0);
                this.Hh = a;
                this.si = c;
                this.Eh = b || !1;
                this.yh = new CXa;
                this.yh.bindTo("position", this);
                this.Ch = new _.Jo(0, 0);
                this.size = new _.No(0, 0);
                this.anchorPoint = new _.Jo(0,
                    0);
                this.Jh = [_.Hn(this, "dragstart", this.oi), _.Hn(this, "dragend", this.di), _.Hn(this, "panbynow", () => {
                    _.pq(this.rh)
                })];
                this.getPosition = _.Ao("position");
                this.getPanes = _.Ao("panes");
                this.getVisible = _.Ao("visible");
                this.getIcon = _.Ao("icon");
                this.getLabel = _.Ao("label")
            }
            LG() {}
            get zn() {
                return this.Bh
            }
            set zn(a) {
                this.Bh !== a && (this.Bh = a, _.Wn(this, "UPDATE_FOCUS"))
            }
            get Nl() {
                return this.get("dragging")
            }
            panes_changed() {
                OR(this);
                this.rh.ri()
            }
            kp(a) {
                this.set("position", a && new _.Jo(a.Sh, a.Th))
            }
            zt() {
                this.unbindAll();
                this.set("panes", null);
                this.oh && this.oh.stop();
                this.uh && (_.Jn(this.uh), this.uh = null);
                this.oh = null;
                MR(this.Jh);
                this.Jh = [];
                OR(this);
                _.Wn(this, "RELEASED")
            }
            Fh() {
                var a;
                if (!(a = this.Xh !== PR(this) || this.bi !== this.getDraggable())) {
                    a = this.Ph;
                    var b = this.get("shape");
                    a = !(a == null || b == null ? a === b : a.type === b.type && _.FI(a.coords, b.coords))
                }
                a && (this.Xh = PR(this), this.bi = this.getDraggable(), this.Ph = this.get("shape"), NR(this), this.rh.ri())
            }
            nh() {
                this.rh.ri()
            }
            position_changed() {
                this.Eh ? _.pq(this.rh) : this.rh.ri()
            }
            xq() {
                return this.targetElement
            }
            Zw() {
                const a =
                    this.xq();
                if (a) {
                    var b = !!this.get("title");
                    b || (b = (b = this.getLabel()) ? !!b.text : !1);
                    this.zn ? a.setAttribute("role", "button") : b ? a.setAttribute("role", "img") : a.removeAttribute("role")
                }
            }
            Ey(a) {
                _.Wn(this, "click", a);
                _.N(window, 171149)
            }
            Ht() {}
            Ss(a) {
                _.Zw(a);
                _.Wn(this, "click", a);
                _.N(window, 171150)
            }
            Dy() {}
            getDraggable() {
                return !!this.get("draggable")
            }
            oi() {
                this.set("dragging", !0);
                this.yh.set("snappingCallback", this.Hh)
            }
            di() {
                this.yh.set("snappingCallback", null);
                this.set("dragging", !1)
            }
            animation_changed() {
                this.zh = !1;
                this.get("animation") ? jWa(this) : (this.set("animating", !1), this.oh && this.oh.stop())
            }
            LF(a) {
                const b = this.get("markerPosition");
                return this.tq && b && this.tq.size ? HVa(a, this.targetElement) : !1
            }
        };
    _.z = vS.prototype;
    _.z.shape_changed = vS.prototype.Fh;
    _.z.clickable_changed = vS.prototype.Fh;
    _.z.draggable_changed = vS.prototype.Fh;
    _.z.cursor_changed = vS.prototype.nh;
    _.z.scale_changed = vS.prototype.nh;
    _.z.raiseOnDrag_changed = vS.prototype.nh;
    _.z.crossOnDrag_changed = vS.prototype.nh;
    _.z.zIndex_changed = vS.prototype.nh;
    _.z.opacity_changed = vS.prototype.nh;
    _.z.title_changed = vS.prototype.nh;
    _.z.cross_changed = vS.prototype.nh;
    _.z.icon_changed = vS.prototype.nh;
    _.z.visible_changed = vS.prototype.nh;
    _.z.dragging_changed = vS.prototype.nh;
    var rWa = "click dblclick mouseup mousedown mouseover mouseout rightclick dragstart drag dragend contextmenu".split(" "),
        CWa = class {
            constructor(a, b, c, d, e, f, g) {
                this.marker = a;
                this.nh = b;
                this.Mh = e;
                this.Ah = f;
                this.ph = g;
                this.xh = !0;
                this.yh = this.zh = null;
                this.rh = [];
                this.wh = b instanceof _.po;
                f = TR(this);
                b = this.wh && f ? _.Cx(f, b.getProjection()) : null;
                this.mh = new vS(d, !!this.wh, h => {
                    this.mh.tq = a.__gm.tq = { ...a.__gm.tq,
                        QQ: h
                    };
                    a.__gm.ny && a.__gm.ny()
                });
                _.Hn(this.mh, "RELEASED", () => {
                    var h = this.mh;
                    if (this.ph && this.ph.has(h)) {
                        ({
                                iF: h
                            } =
                            this.ph.get(h));
                        for (const k of h) k.remove()
                    }
                    this.ph && this.ph.delete(this.mh)
                });
                this.Ah && this.ph && !this.ph.has(this.mh) && (this.ph.set(this.mh, {
                    marker: this.marker,
                    iF: []
                }), this.Ah.Ah(this.mh), UR(this, this.mh), oWa(this, this.mh));
                (this.qh = this.wh ? new _.lN(e.wk, this.mh, b, e, () => {
                    if (this.mh.get("dragging") && !this.marker.get("place")) {
                        var h = this.qh.getPosition();
                        h && (h = _.Ar(h, this.nh.get("projection")), this.xh = !1, this.marker.set("position", h), this.xh = !0)
                    }
                }) : null) && e.Cj(this.qh);
                this.sh = new EXa(c, (h, k, n) => {
                    this.mh.tq =
                        a.__gm.tq = { ...a.__gm.tq,
                            size: h,
                            anchor: k,
                            labelOrigin: n
                        };
                    a.__gm.ny && a.__gm.ny()
                });
                this.oh = this.wh ? null : new _.$M;
                this.th = this.wh ? null : new FXa;
                this.uh = new _.$n;
                this.uh.bindTo("position", this.marker);
                this.uh.bindTo("place", this.marker);
                this.uh.bindTo("draggable", this.marker);
                this.uh.bindTo("dragging", this.marker);
                this.sh.bindTo("modelIcon", this.marker, "icon");
                this.sh.bindTo("modelLabel", this.marker, "label");
                this.sh.bindTo("modelCross", this.marker, "cross");
                this.sh.bindTo("modelShape", this.marker, "shape");
                this.sh.bindTo("useDefaults", this.marker, "useDefaults");
                this.mh.bindTo("icon", this.sh, "viewIcon");
                this.mh.bindTo("label", this.sh, "viewLabel");
                this.mh.bindTo("cross", this.sh, "viewCross");
                this.mh.bindTo("shape", this.sh, "viewShape");
                this.mh.bindTo("title", this.marker);
                this.mh.bindTo("cursor", this.marker);
                this.mh.bindTo("dragging", this.marker);
                this.mh.bindTo("clickable", this.marker);
                this.mh.bindTo("zIndex", this.marker);
                this.mh.bindTo("opacity", this.marker);
                this.mh.bindTo("anchorPoint", this.marker);
                this.mh.bindTo("markerPosition",
                    this.marker, "position");
                this.mh.bindTo("animation", this.marker);
                this.mh.bindTo("crossOnDrag", this.marker);
                this.mh.bindTo("raiseOnDrag", this.marker);
                this.mh.bindTo("animating", this.marker);
                this.th || this.mh.bindTo("visible", this.marker);
                pWa(this);
                qWa(this);
                sWa(this);
                this.wh ? (tWa(this), uWa(this), wWa(this)) : (xWa(this), this.oh && (this.th.bindTo("visible", this.marker), this.th.bindTo("cursor", this.marker), this.th.bindTo("icon", this.marker), this.th.bindTo("icon", this.sh, "viewIcon"), this.th.bindTo("mapPixelBoundsQ",
                    this.nh.__gm, "pixelBoundsQ"), this.th.bindTo("position", this.oh, "pixelPosition"), this.mh.bindTo("visible", this.th, "shouldRender")), zWa(this))
            }
            dispose() {
                this.mh.set("animation", null);
                this.mh.zt();
                this.Mh && this.qh ? this.Mh.Ql(this.qh) : this.mh.zt();
                this.th && this.th.unbindAll();
                this.oh && this.oh.unbindAll();
                this.sh.unbindAll();
                this.uh.unbindAll();
                this.rh.forEach(_.Jn);
                this.rh.length = 0
            }
        };
    var $R = class {
        constructor(a, b, c, d) {
            this.div = a;
            this.zj = b;
            this.mh = c;
            this.fi = d
        }
        getContext() {
            if (!this.context) {
                const a = this.div,
                    b = a.ownerDocument.createElement("canvas");
                _.Hq(b);
                b.style.position = "absolute";
                b.style.top = b.style.left = "0";
                const c = b.getContext("2d"),
                    d = WR(c),
                    e = this.fi.size;
                b.width = Math.ceil(e.Sh * d);
                b.height = Math.ceil(e.Th * d);
                b.style.width = _.Em(e.Sh);
                b.style.height = _.Em(e.Th);
                a.appendChild(b);
                this.context = c
            }
            return this.context
        }
        cE(a) {
            const b = EWa(this),
                c = this.getContext(),
                d = WR(c),
                e = Math.round(a.dx *
                    d),
                f = Math.round(a.dy * d),
                g = Math.ceil(a.xr * d);
            a = Math.ceil(a.ur * d);
            const h = DWa(this, g, a),
                k = h.getContext("2d");
            k.translate(-e, -f);
            b.forEach(n => {
                k.globalAlpha = _.ym(n.opacity, 1);
                k.drawImage(n.image, n.Dz, n.Ez, n.Cz, n.wz, Math.round(n.dx * d), Math.round(n.dy * d), n.xr * d, n.ur * d)
            });
            c.clearRect(e, f, g, a);
            c.globalAlpha = 1;
            c.drawImage(h, e, f)
        }
    };
    $R.prototype.PM = $R.prototype.cE;
    var NWa = class {
        constructor() {
            this.mh = _.VI().mh
        }
        load(a, b) {
            return this.mh.load(new _.uL(a.url), c => {
                if (c) {
                    var d = c.size,
                        e = a.size || a.scaledSize || d;
                    a.size = e;
                    var f = a.scaledSize || d,
                        g = a.anchor || new _.Jo(e.width / 2, e.height),
                        h = f.width / d.width,
                        k = f.height / d.height,
                        n = a.origin ? a.origin.x / h : 0,
                        p = a.origin ? a.origin.y / k : 0,
                        r = -g.x;
                    g = -g.y;
                    var t = e.width / h,
                        v = e.width,
                        w = e.height / k,
                        y = e.height;
                    n * h + e.width > f.width && (t = d.width - n * h, v = f.width);
                    p * k + e.height > f.height && (w = d.height - p * k, y = f.height);
                    b({
                        image: c,
                        Dz: n,
                        Ez: p,
                        Cz: t,
                        wz: w,
                        dx: r,
                        dy: g,
                        xr: v,
                        ur: y
                    })
                } else b(null)
            })
        }
        cancel(a) {
            this.mh.cancel(a)
        }
    };
    var PWa = class {
        constructor(a, b, c, d) {
            this.mh = b;
            this.nh = c;
            this.zIndex = 40;
            this.oh = new _.mN(a, d, c)
        }
        Ft(a) {
            return a !== "dragstart" && a !== "drag" && a !== "dragend"
        }
        Nt(a, b) {
            return b ? YR(this, a, -8, 0) || YR(this, a, 0, -8) || YR(this, a, 8, 0) || YR(this, a, 0, 8) : YR(this, a, 0, 0)
        }
        handleEvent(a, b, c) {
            const d = b.jk;
            if (a === "mouseout") this.mh.set("cursor", ""), this.mh.set("title", null);
            else if (a === "mouseover") {
                var e = d.qw;
                this.mh.set("cursor", e.cursor);
                (e = e.title) && this.mh.set("title", e)
            }
            let f;
            d && a !== "mouseout" ? f = d.qw.latLng : f = b.latLng;
            a ===
                "dblclick" && _.Fn(b.domEvent);
            _.Wn(c, a, new _.MB(f, b.domEvent))
        }
    };
    var QWa = class extends _.mr {
        constructor(a, b, c, d, e, f, g) {
            super();
            this.tiles = a;
            this.qh = d;
            this.oh = c;
            this.nh = e;
            this.ph = f;
            this.fi = g || _.XB;
            b.mh = h => {
                IWa(this, h)
            };
            b.onRemove = h => {
                JWa(this, h)
            };
            b.forEach(h => {
                IWa(this, h)
            })
        }
        mh() {
            return {
                fi: this.fi,
                sm: 2,
                vl: this.vl.bind(this)
            }
        }
        vl(a, b = {}) {
            const c = document.createElement("div"),
                d = this.fi.size;
            c.style.width = `${d.Sh}px`;
            c.style.height = `${d.Th}px`;
            c.style.overflow = "hidden";
            a = {
                div: c,
                zoom: a.ii,
                ej: new _.Jo(a.Zh, a.ai),
                bq: {},
                zj: new _.jr
            };
            c.fl = a;
            KWa(this, a);
            let e = !1;
            return {
                Pj: () =>
                    c,
                ln: () => e,
                loaded: new Promise(f => {
                    _.Sn(c, "load", () => {
                        e = !0;
                        f()
                    })
                }),
                release: () => {
                    const f = c.fl;
                    c.fl = null;
                    LWa(this, f);
                    c.textContent = "";
                    b.Zj && b.Zj()
                }
            }
        }
    };
    var RWa = class {
        constructor(a, b, c) {
            this.nh = b;
            this.timeout = null;
            this.mh = !1;
            this.oh = 0;
            this.zj = a;
            const d = this;
            a.mh = e => {
                d.Rq(e, !0)
            };
            a.onRemove = e => {
                d.onRemove(e)
            };
            this.div = c;
            a.getSize() ? (this.mh = !0, this.fp()) : queueMicrotask(() => {
                _.Wn(c, "load")
            })
        }
        onRemove(a) {
            this.Rq(a, !1)
        }
        Rq(a, b) {
            this.oh++ < 4 ? b ? this.nh.cE(a) : this.nh.PM(a) : this.mh = !0;
            this.timeout || (this.timeout = _.XI(this.fp.bind(this)))
        }
        fp() {
            this.mh && FWa(this.nh);
            this.mh = !1;
            this.timeout = null;
            this.oh = 0;
            queueMicrotask(() => {
                _.Wn(this.div, "load")
            })
        }
    };
    var OWa = class {
        constructor(a, b, c, d, e) {
            var f = MWa;
            this.markers = a;
            this.nh = b;
            this.mh = c;
            this.qh = f;
            this.ph = d;
            this.oh = e;
            this.markers.mh = g => {
                this.Cu(g)
            };
            this.markers.onRemove = g => {
                this.Cw(g)
            }
        }
        Cu(a) {
            var b = a.get("internalPosition"),
                c = a.get("zIndex");
            const d = a.get("opacity"),
                e = a.__gm.Jy = {
                    KA: a,
                    latLng: b,
                    zIndex: c,
                    opacity: d,
                    zj: {}
                };
            b = a.get("useDefaults");
            c = a.get("icon");
            const f = a.get("shape") || c && !b ? a.get("shape") : this.mh.shape,
                g = c ? this.qh(c) : this.mh.icon,
                h = GVa(() => {
                    e === a.__gm.Jy && (e.su || e.mH) && this.vm(a, e, g, f)
                });
            g.url ? this.ph.load(g, k => {
                e.su = k;
                h()
            }) : (e.mH = this.oh(g), h())
        }
        Cw(a) {
            this.nh.remove(a.__gm.Jy);
            a.__gm.Jy = null
        }
        vm(a, b, c, d) {
            if (b.su) {
                c = c.size;
                var e = a.get("anchorPoint");
                if (!e || e.mh) e = new _.Jo(b.su.dx + c.width / 2, b.su.dy), e.mh = !0, a.set("anchorPoint", e)
            } else c = b.mH.size;
            d ? d.coords = d.coords || d.coord : d = {
                type: "rect",
                coords: [0, 0, c.width, c.height]
            };
            b.shape = d;
            b.clickable = a.get("clickable");
            b.title = a.get("title") || null;
            b.cursor = a.get("cursor") || "pointer";
            _.rq(this.nh, b)
        }
    };
    var ZR = new Map;
    var HXa = class {
        constructor(a, b, c, d) {
            this.Ev = {};
            this.Qt = 0;
            this.Lw = !0;
            const e = this;
            this.eD = b;
            this.hu = c;
            this.jF = d;
            const f = {
                animating: 1,
                animation: 1,
                attribution: 1,
                clickable: 1,
                cursor: 1,
                draggable: 1,
                flat: 1,
                icon: 1,
                label: 1,
                opacity: 1,
                optimized: 1,
                place: 1,
                position: 1,
                shape: 1,
                __gmHiddenByCollision: 1,
                title: 1,
                visible: 1,
                zIndex: 1
            };
            this.nM = function(g) {
                g in f && (delete this.changed, e.Ev[_.Zn(this)] = this, UWa(e))
            };
            a.mh = g => {
                e.Cu(g)
            };
            a.onRemove = g => {
                e.Cw(g)
            };
            a = a.elements;
            for (const g of Object.values(a)) this.Cu(g)
        }
        Cu(a) {
            this.Ev[_.Zn(a)] =
                a;
            UWa(this)
        }
        Cw(a) {
            delete a.changed;
            delete this.Ev[_.Zn(a)];
            this.eD.remove(a);
            this.hu.remove(a)
        }
    };
    var oS = class extends Event {
        constructor() {
            super("gmp-click", {
                bubbles: !0,
                composed: !0
            })
        }
    };
    var IXa = class {
        Ch() {}
        zh() {}
        nh() {}
        oh() {}
        ph() {}
        uh() {}
        xh() {}
        sh() {}
        qh() {}
        rh() {}
        wh() {}
        yh() {}
        mh() {}
        Ah() {}
        Bh() {}
        Fh() {}
        Eh() {}
        th() {}
    };
    var JXa = (0, _.bj)
    `.yNHHyP-marker-view .IPAZAH-content-container\u003e*{pointer-events:none}.yNHHyP-marker-view .IPAZAH-content-container.HJDHPx-interactive\u003e*{pointer-events:auto}\n`;
    var YWa = CVa("visible-gmp-advanced-markers"),
        XWa = CVa("hidden-gmp-advanced-markers"),
        aXa = class {
            constructor(a) {
                this.ki = KXa;
                this.sp = null;
                this.wh = !1;
                this.th = null;
                this.sh = 0;
                this.uh = null;
                this.map = a;
                this.ph = new Set;
                this.qh = new Set;
                this.Xn = `maps-aria-${_.oo()}`;
                this.oh = document.createElement("span");
                this.oh.id = this.Xn;
                this.oh.textContent = "To activate drag with keyboard, press Alt + Enter. Once in keyboard drag state, use the arrow keys to move the marker. To complete the drag, press the Enter key. To cancel, press Escape.";
                this.oh.style.display = "none";
                this.rh = document.createElement("div");
                this.mh = document.createElement("div");
                CSS.supports("content-visibility: hidden") ? this.mh.style.contentVisibility = "hidden" : this.mh.style.visibility = "hidden";
                var b = document.createElement("slot");
                b.setAttribute("name", YWa);
                this.rh.appendChild(b);
                b = document.createElement("slot");
                b.setAttribute("name", XWa);
                this.mh.appendChild(b);
                this.nh = document.createElement("div");
                this.nh.append(this.rh, this.mh);
                const c = a.__gm;
                this.yh = c.uq;
                this.xh = new Promise(d => {
                    c.oh.then(e => {
                        this.map && (e && (this.sp = VWa(this, a)), this.wh = !0);
                        d()
                    })
                });
                _.Mu(JXa, this.map.getDiv());
                Promise.all([c.nh, this.xh]).then(([{
                    xm: d
                }]) => {
                    this.map && d.overlayMouseTarget.append(this.oh, this.nh);
                    this.uh = c.addListener("panes_changed", e => {
                        this.map && e.overlayMouseTarget.append(this.oh, this.nh)
                    })
                })
            }
            dispose() {
                this.sp && (this.sp.setMap(null), this.sp = null);
                this.uh && this.uh.remove();
                this.oh.remove();
                this.mh.remove();
                this.rh.remove();
                this.nh.remove();
                this.mh.textContent = "";
                this.rh.textContent = "";
                this.ph.clear();
                this.qh.clear();
                this.map = null
            }
            isEmpty() {
                return this.ph.size === 0
            }
            requestRedraw() {
                this.wh ? this.sp && this.sp.requestRedraw() : this.xh.then(() => {
                    this.sp && this.sp.requestRedraw()
                })
            }
            onDraw(a) {
                if (this.map) {
                    var b = this.yh.offsetWidth,
                        c = this.yh.offsetHeight,
                        d = _.Nq(this.map.getZoom() || 1, this.map.getTilt() || 0, this.map.getHeading() || 0);
                    for (const h of this.ph.values()) {
                        a: {
                            var e = h.IL;
                            var f = this.map.getCenter();
                            if (!e || !f) {
                                e = null;
                                break a
                            }
                            f = _.tm(f.lng(), -180, 180);
                            var g = _.tm(e.lng, -180, 180);f > 0 && g < f - 180 ? g += 360 : f < 0 && g >
                                f + 180 && (g -= 360);e = new _.Ep({
                                altitude: e.altitude,
                                lat: e.lat,
                                lng: g
                            }, !0)
                        }
                        if (!e) {
                            h.kp(null, d);
                            continue
                        }
                        e = a.fromLatLngAltitude(e);f = Array.from(e);e = g = [0, 0, 0];
                        const k = e[0],
                            n = e[1],
                            p = e[2],
                            r = 1 / (f[3] * k + f[7] * n + f[11] * p + f[15]);e[0] = (f[0] * k + f[4] * n + f[8] * p + f[12]) * r;e[1] = (f[1] * k + f[5] * n + f[9] * p + f[13]) * r;e[2] = (f[2] * k + f[6] * n + f[10] * p + f[14]) * r;
                        const {
                            zL: t,
                            gO: v
                        } = {
                            zL: f[14] < 0 && f[15] < 0,
                            gO: g
                        };t ? h.kp(null, d) : h.kp({
                            Sh: FR(v[0] / 2 * b),
                            Th: FR(-v[1] / 2 * c)
                        }, d, {
                            Sh: b,
                            Th: c
                        })
                    }
                }
            }
        };
    var cS = new Map,
        KXa = new class extends IXa {
            Ch(a) {
                a && a && _.N(a, 181191)
            }
            zh(a) {
                if (a) {
                    var b = a.getRenderingType();
                    b !== "UNINITIALIZED" && a && _.N(a, 159713);
                    b === "RASTER" ? a && _.N(a, 157416) : b === "VECTOR" && a && _.N(a, 157417)
                }
            }
            nh(a, b = !1) {
                a && _.N(a, 158896);
                b && a && _.N(a, 185214)
            }
            oh(a, b) {
                b && (b !== "REQUIRED" && a && _.N(a, 160097), b === "REQUIRED_AND_HIDES_OPTIONAL" ? a && _.N(a, 160098) : b === "OPTIONAL_AND_HIDES_LOWER_PRIORITY" && a && _.N(a, 160099))
            }
            ph(a, b) {
                b ? a && _.N(a, 159404) : a && _.N(a, 159405)
            }
            uh(a) {
                a && _.N(a, 159484)
            }
            xh(a) {
                a && _.N(a, 160438)
            }
            sh(a) {
                a &&
                    _.N(a, 159521)
            }
            qh(a) {
                a && _.N(a, 160103)
            }
            rh(a) {
                a && _.N(a, 159805)
            }
            wh(a) {
                a && _.N(a, 159490)
            }
            yh(a) {
                a && _.N(a, 159812)
            }
            mh(a) {
                a && _.N(a, 159609)
            }
            Ah(a) {
                a && _.N(a, 160122)
            }
            Bh(a) {
                a && _.N(a, 160106)
            }
            Fh(a) {
                a && _.N(a, 160478)
            }
            Eh(a, b) {
                const c = [{
                    threshold: 1E4,
                    Hp: 160636
                }, {
                    threshold: 5E3,
                    Hp: 160635
                }, {
                    threshold: 2E3,
                    Hp: 160634
                }, {
                    threshold: 1E3,
                    Hp: 160633
                }, {
                    threshold: 500,
                    Hp: 160632
                }, {
                    threshold: 200,
                    Hp: 160631
                }, {
                    threshold: 100,
                    Hp: 160630
                }, {
                    threshold: 50,
                    Hp: 159732
                }, {
                    threshold: 10,
                    Hp: 160629
                }, {
                    threshold: 1,
                    Hp: 160628
                }];
                for (const {
                        threshold: d,
                        Hp: e
                    } of c)
                    if (b >=
                        d) {
                        a && _.N(a, e);
                        break
                    }
            }
            th(a) {
                window && _.N(window, a instanceof KeyboardEvent ? 171152 : 171153)
            }
        },
        LXa = new IXa,
        bS = null;
    var MXa = class {
        constructor(a) {
            this.mh = a;
            this.qh = this.ph = !1;
            this.uh = this.nh = this.rh = this.wh = this.yh = this.Eh = null;
            this.xh = 0;
            this.Hh = null;
            this.Nh = b => {
                this.Gt(b)
            };
            this.Ph = b => {
                b.touches.length === 1 && this.Gt(b)
            };
            this.Jh = b => {
                b.preventDefault();
                b.stopImmediatePropagation()
            };
            this.Bh = b => {
                if (this.qh || this.th || MVa(b, this.Eh)) this.th = !0
            };
            a = this.mh.Ep;
            _.iy !== 2 ? (a.addEventListener("pointerdown", this.Nh), a.addEventListener("pointermove", this.Bh)) : (a.addEventListener("touchstart", this.Ph, {
                passive: !1
            }), a.addEventListener("touchmove",
                this.Bh, {
                    passive: !1
                }));
            a.addEventListener("mousedown", this.Jh);
            this.Ah = b => {
                b.preventDefault();
                b.stopImmediatePropagation();
                this.qh ? mXa(this, b) : this.ph ? (nXa(this, b), gS(this.mh, "drag", b)) : (oXa(this, b), b = this.mh, b.ki.Fh(b.map))
            };
            this.sh = b => {
                this.uh && b.timeStamp - this.uh >= 500 && (!this.ph || this.qh) ? (this.qh ? mXa(this, b) : (oXa(this, b), b = this.mh, b.ki.Bh(b.map), b.Pp && _.Wn(b, "longpressdragstart")), this.th = !0) : (this.ph && (this.qh || this.th || MVa(b, this.Eh)) && (this.th = !0), this.qh && dS(this, b), b.type === "touchend" && hS(this, !1), this.ph ? (b.stopImmediatePropagation(), nXa(this, b), kS(this), lS(this.mh, !0), gS(this.mh, "dragend", b)) : kS(this))
            };
            this.Vh = b => {
                this.bi(b)
            };
            this.Xh = b => {
                this.di(b)
            };
            this.Oh = b => {
                eS(this, b)
            };
            this.bi = b => {
                if (b.altKey && (_.tz(b) || b.key === _.Gpa)) eS(this, b);
                else if (!b.altKey && _.tz(b)) this.th = !0, dS(this, b);
                else if (_.uz(b) || _.wz(b) || _.vz(b) || _.xz(b)) b.preventDefault(), this.zh.add(b.key), this.xh || (this.Hh = new _.zM(100), qXa(this)), gS(this.mh, "drag", b);
                else if (b.code === "Equal" || b.code === "Minus") {
                    var c = this.mh;
                    b =
                        b.code === "Equal" ? 1 : -1;
                    const d = LVa(c.Mk, c.Sp);
                    d && c.Mh.NH(b, d)
                }
            };
            this.di = b => {
                (_.uz(b) || _.wz(b) || _.vz(b) || _.xz(b)) && this.zh.delete(b.key)
            };
            this.Ch = () => {
                hS(this, !0)
            };
            this.Fh = () => {
                this.ph || hS(this, !1)
            };
            this.oh = document.createElement("div");
            iXa(this);
            this.th = !1;
            this.zh = new Set
        }
        Kz(a) {
            this.nh && _.AM(this.nh, a)
        }
        Gt(a) {
            this.th = !1;
            if (this.mh.gmpDraggable && (a.button === 0 || a.type === "touchstart")) {
                const b = this.mh.Ep;
                b.focus();
                const c = document;
                _.iy !== 2 || a.preventDefault();
                a.stopImmediatePropagation();
                this.uh = a.timeStamp;
                _.iy !== 2 ? (c.addEventListener("pointermove", this.Ah), c.addEventListener("pointerup", this.sh), c.addEventListener("pointercancel", this.sh)) : (c.addEventListener("touchmove", this.Ah, {
                    passive: !1
                }), c.addEventListener("touchend", this.sh), c.addEventListener("touchcancel", this.sh), c.addEventListener("touchstart", this.sh));
                this.ph || (this.Eh = _.ZL(a));
                b.style.cursor = _.Bz
            }
        }
        Ey() {
            this.ph || (this.th = !1)
        }
        Ht(a) {
            if (this.mh.gmpDraggable && !this.qh && !this.ph) {
                var b = this.mh.Ep;
                b.addEventListener("keydown", this.Vh);
                b.addEventListener("keyup",
                    this.Xh);
                b.addEventListener("blur", this.Oh);
                this.rh = this.mh.ho();
                this.yh = this.mh.position;
                this.qh = this.ph = !0;
                lXa(this);
                b = this.mh.Ep;
                b.setAttribute("aria-grabbed", "true");
                jS(this.mh);
                b.style.zIndex = "2147483647";
                this.oh.style.opacity = "1";
                gS(this.mh, "dragstart", a);
                a = this.mh;
                a.ki.Ah(a.map)
            }
        }
        Dy(a, b = !0) {
            this.qh ? eS(this, a, b) : this.ph && (this.mh.position = this.yh, a.stopImmediatePropagation(), kS(this), b && gS(this.mh, "dragend", a))
        }
        Nl() {
            return this.ph
        }
        dispose() {
            kS(this);
            const a = this.mh.Ep;
            _.iy !== 2 ? (a.removeEventListener("pointerdown",
                this.Nh), a.removeEventListener("pointermove", this.Bh)) : (a.removeEventListener("touchstart", this.Ph, {
                passive: !1
            }), a.removeEventListener("touchmove", this.Bh, {
                passive: !1
            }));
            a.removeEventListener("mousedown", this.Jh);
            a.removeEventListener("pointerenter", this.Ch);
            a.removeEventListener("pointerleave", this.Fh);
            a.removeEventListener("focus", this.Ch);
            a.removeEventListener("blur", this.Fh);
            this.oh.remove()
        }
    };
    var wS = !1,
        xS = class extends _.St {
            set anchorTop(a) {
                this.vL = this.Kh("anchorTop", _.Ts, a);
                sS(this, this.Rk);
                lS(this, !0)
            }
            get anchorTop() {
                return this.vL
            }
            set anchorLeft(a) {
                this.uL = this.Kh("anchorLeft", _.Ts, a);
                sS(this, this.Rk);
                lS(this, !0)
            }
            get anchorLeft() {
                return this.uL
            }
            constructor(a = {}) {
                super(a);
                this.Bv = this.xj = this.Hj = null;
                this.GA = "";
                this.Pl = this.lw = this.MD = this.Mh = this.Kk = this.kl = null;
                this.JD = !1;
                this.Oz = null;
                this.vC = this.OF = this.Pz = this.LD = !1;
                this.ui = this.zx = this.jG = this.IG = this.OH = this.rE = null;
                this.ID =
                    void 0;
                this.av = this.KD = !1;
                this.xF = _.Zo(!1);
                this.Sp = this.bv = this.VA = null;
                this.ls = "";
                this.Mk = this.Qz = void 0;
                this.sA = this.tA = !0;
                this.EB = this.YC = !1;
                this.uw = !0;
                document.createElement("div");
                sXa(this);
                this.targetElement = this;
                this.Ep = this;
                this.Pp = wS;
                Object.defineProperties(this, {
                    Pp: {
                        value: wS,
                        writable: !1
                    }
                });
                this.ki = this.Pp ? LXa : KXa;
                this.addEventListener("focus", e => {
                    this.jB(e)
                }, !0);
                this.addEventListener("marker-resize", () => {
                    tXa(this);
                    lS(this, !0)
                });
                this.pm = new _.iC;
                this.Gj = document.createElement("div");
                _.Gr(this.Gj,
                    "content-container");
                this.kl = document.createElement("slot");
                this.kl.addEventListener("slotchange", () => {
                    this.Ts()
                });
                this.Gj.appendChild(this.kl);
                this.kl.prepend(this.pm);
                this.Ts();
                Promise.resolve().then(() => {
                    mS(this);
                    a.anchorTop && (this.anchorTop = a.anchorTop);
                    a.anchorLeft && (this.anchorLeft = a.anchorLeft)
                });
                this.Wx = getComputedStyle(this);
                this.IK = (e, f, g) => this.ty(e, f, g);
                const b = () => {
                        mS(this);
                        nS(this);
                        const e = _.In(this, "gmp-click");
                        this.ki.nh(this.map, e)
                    },
                    c = () => {
                        mS(this);
                        nS(this)
                    },
                    d = ["click"];
                d.push("gmp-click");
                for (const e of d) FVa(this, e, b), EVa(this, e, c);
                this.kn = new _.$n;
                this.kn.addListener("isopen_changed", () => {
                    uXa(this);
                    tXa(this)
                });
                this.mF = e => {
                    this.xj ? .Yr(e === "cooperative" || e === "none")
                };
                this.collisionBehavior = a.collisionBehavior;
                a.content != null && (this.content = a.content);
                this.DA = !!a.DA;
                this.gmpClickable = a.gmpClickable;
                this.gmpDraggable = a.gmpDraggable;
                this.position = a.position;
                this.title = a.title ? ? "";
                this.zIndex = a.zIndex;
                this.map = a.map;
                this.yi(a, xS, "AdvancedMarkerElement")
            }
            addEventListener(a, b, c) {
                a !== "click" ||
                    this.YC || _.Bn(_.Wp(this, "Please use addEventListener('gmp-click', ...) instead of addEventListener('click', ...)."));
                super.addEventListener(a, b, c)
            }
            addListener(a, b) {
                a === "click" && _.Bn(_.Wp(this, "Please use addEventListener('gmp-click', ...) instead of addEventListener('click', ...)."));
                return _.Hn(this, a, b)
            }
            jB(a) {
                var b = a.target,
                    c = a.relatedTarget;
                if (this !== b)
                    if (a.stopPropagation(), a.stopImmediatePropagation(), console.debug('Focusable child elements in AdvancedMarkerElement are not supported. To make AdvancedMarkerElement focusable, use addListener() to register a "click" event on the AdvancedMarkerElement instance.'),
                        this.ki.yh(this.map), a = [document.body, ..._.yx(document.body)], b = a.indexOf(b), c = a.indexOf(c), b === -1 || c === -1) this.focus();
                    else
                        for (c = b > c ? 1 : -1, b += c; b >= 0 && b < a.length; b += c) {
                            const d = a[b];
                            if (this.zn && d === this || !this.contains(d)) {
                                (d instanceof HTMLElement || d instanceof SVGElement) && d.focus();
                                break
                            }
                        }
            }
            Ey(a) {
                this.Hj && this.Hj.Ey();
                vXa(this, a)
            }
            Ht(a) {
                this.Hj && this.Hj.Ht(a)
            }
            Gt(a) {
                this.Hj && this.Hj.Gt(a)
            }
            hF() {
                return new Promise(a => {
                    if (this.zn) {
                        var b = () => {
                            this.isConnected && this.Rk ? setTimeout(() => {
                                    this.focus();
                                    a()
                                },
                                0) : _.rz(_.qz(), b)
                        };
                        b()
                    }
                })
            }
            Ss() {}
            Dy(a) {
                this.Hj && (this.Hj.Dy(a, !this.Pp), this.Pp && _.Wn(this, "dragcancel"))
            }
            get collisionBehavior() {
                return this.ID
            }
            set collisionBehavior(a) {
                a = this.Kh("collisionBehavior", _.cn(_.Wm(_.ht)), a) || "REQUIRED";
                this.collisionBehavior !== a && (this.ID = a, this.ki.oh(this.map, this.ID), this.map && (!aS(this) && this.ui ? AVa(this.ui.Hh, this) : lS(this, !0)))
            }
            get element() {
                return this
            }
            get Qy() {
                return pS(this)[0] === this.pm
            }
            get content() {
                const a = pS(this);
                a.length > 1 && console.debug("The content getter of AdvancedMarkerElement only returns the first content when there are multiple contents, use childNodes or children to get all the contents.");
                return a[0]
            }
            set content(a) {
                a = this.Kh("content", _.cn(_.an([_.Vm(Node, "Node"), _.$m(_.Um)])), a) || this.pm;
                const b = pS(this);
                if (b.length !== 1 || b[0] !== a) a !== this.pm ? this.replaceChildren(a) : a === this.pm && this.replaceChildren(), this.Ts()
            }
            Ts() {
                const a = () => {
                    this.kl && !this.kl.contains(this.pm) && this.kl.prepend(this.pm);
                    this.Pl = null;
                    this.Hj && iS(this.Hj);
                    lS(this, !0);
                    mS(this)
                };
                this.ki.ph(this.map, this.Qy);
                this.cq.then(() => {
                    a()
                })
            }
            get dragIndicator() {}
            set dragIndicator(a) {}
            get gmpClickable() {
                return this.KD
            }
            set gmpClickable(a) {
                a =
                    this.Kh("gmpClickable", _.Us, a) || !1;
                this.KD !== a && (this.KD = a, mS(this), nS(this))
            }
            get gmpDraggable() {
                return this.av
            }
            set gmpDraggable(a) {
                a = this.Kh("gmpDraggable", _.Us, a) || !1;
                xXa(this, this.position, a);
                this.xF.set(a);
                this.av !== a && ((this.av = a) ? (this.ki.qh(this.map), this.setAttribute("aria-grabbed", "false"), bXa(this, this.GA), this.Hj = new MXa(this), hXa(this.Hj)) : (this.removeAttribute("aria-grabbed"), this.LG(this.GA), this.Hj.dispose(), this.Hj = null), mS(this), nS(this))
            }
            LG(a) {
                var b = this.getAttribute("aria-describedby");
                b = (b ? b.split(" ") : []).filter(c => c !== a);
                b.length > 0 ? this.setAttribute("aria-describedby", b.join(" ")) : this.removeAttribute("aria-describedby")
            }
            get map() {
                return this.Mk
            }
            set map(a) {
                this.setMap(a);
                this.Mk && (a = _.qo(this.Mk)) && this.parentElement !== a && a.append(this)
            }
            setMap(a) {
                this.Mk !== a && (a = this.Kh("map", _.cn(_.Vm(_.po, "MapsApiMap")), a), a instanceof _.po && (a = a.qN), a && this.isConnected ? qS(this) : this.dispose(), this.Mk = a, this.kn.set("map", this.Mk), this.Mk instanceof _.po ? (this.Mk && cXa(this, this.Mk), this.ui =
                    this.Mk.__gm, wXa(this), this.rE = this.Mk.addListener("bounds_changed", () => {
                        rS(this)
                    }), this.OH = this.Mk.addListener("zoom_changed", () => {
                        rS(this)
                    }), this.IG = this.Mk.addListener("projection_changed", () => {
                        rS(this)
                    }), this.jG = this.Mk.addListener("maptypeid_changed", () => {
                        zXa(this)
                    }), Promise.all([this.ui.nh, this.ui.oh]).then(([b, c]) => {
                        this.Mk === b.map && (this.OF = c, this.ki.zh(b.map), c = this.ui.mh, this.Pp || _.aq(c, "ADVANCED_MARKERS").isAvailable) && (this.Mh = b.Mh, zXa(this))
                    }), AXa(this), BXa(this)) : this.ui = null)
            }
            get position() {
                return this.bv
            }
            set position(a) {
                a =
                    (a = this.Kh("position", _.cn(_.wr), a) || null) && new _.Ep(a);
                const b = this.bv;
                xXa(this, a, this.gmpDraggable);
                if (b && a) {
                    var c = new _.Ep(b, !0);
                    const d = new _.Ep(a, !0);
                    c = !c.equals(d)
                } else c = b !== a;
                c && (this.Sp = (this.bv = a) ? new _.pn(a) : null, this.vC = !0, a || sS(this, null), this.kn.set("position", this.Sp), this.kn.set("map", this.map), this.zx ? dXa(this.map) : yXa(this), this.Il() > 0 && this.ki.mh(this.map), _.Rp(this, "position", b))
            }
            get IL() {
                return this.bv
            }
            get title() {
                return this.ls
            }
            set title(a) {
                a = this.Kh("title", _.Or, a);
                const b =
                    this.ls;
                a !== this.title && (this.ls = a, this.title && this.ki.rh(this.map), this.title === "" ? (this.removeAttribute("aria-label"), this.removeAttribute("title")) : (this.setAttribute("aria-label", this.title), this.setAttribute("title", this.title)), this.Zw(), _.Rp(this, "title", b))
            }
            get zIndex() {
                return this.Qz
            }
            set zIndex(a) {
                a = this.Kh("zIndex", _.cn(_.Os), a);
                this.Qz = a == null ? null : a;
                this.style.zIndex = this.Qz == null ? "" : `${this.Qz}`;
                this.zIndex !== null && this.ki.sh(this.map);
                lS(this)
            }
            get cw() {
                const a = _.In(this, "click"),
                    b = _.In(this,
                        "gmp-click");
                return a || b || !!this.gmpClickable
            }
            get MF() {
                return this.cw || !!this.gmpDraggable
            }
            get zn() {
                return this.JD
            }
            set zn(a) {
                rXa(this);
                this.JD !== a && (this.JD = a, rS(this))
            }
            get Zs() {
                return this.Pz
            }
            set Zs(a) {
                if (a !== this.Pz) {
                    if (this.Pz = a) this.sA = this.tA = !1, jS(this);
                    this.map && eXa(this, this.map)
                }
            }
            get Wo() {
                return this.Oz
            }
            set Wo(a) {
                a !== this.Oz && (this.Oz = a, this.map && eXa(this, this.map), rS(this), _.Wn(this, "UPDATE_BASEMAP_COLLISION"))
            }
            Uv() {
                if (!this.Rk) return null;
                if (!this.Pl)
                    for (const c of pS(this)) {
                        var a = this.Wx;
                        const {
                            offset: d,
                            size: e
                        } = JVa(this, c);
                        var b = ER(a);
                        a = b.offsetY + d.y;
                        b = b.offsetX + d.x;
                        a = _.sp(b, a, b + e.width, a + e.height);
                        this.Pl ? this.Pl.extendByBounds(a) : this.Pl = a
                    }
                return this.Pl
            }
            get Rk() {
                return this.MD
            }
            set Rk(a) {
                const b = this.MD == null && !!a;
                this.MD = a;
                uXa(this, b)
            }
            Il() {
                return this.bv ? this.bv.altitude : 0
            }
            ty(a, b, c) {
                return this.Mk ? (c = _.qEa(this.Mk.getProjection(), this.Sp, c)) ? a / c * Math.sin(b * Math.PI / 180) : 0 : 0
            }
            kp(a, b, c) {
                if (a) {
                    if (this.Hj) {
                        b = this.Hj;
                        var d = b.mh;
                        b = (d = d.ui ? d.ui.uq : null) && b.rh && b.ph && !b.qh ? KVa(d, b.rh) : null
                    } else b =
                        null;
                    b && (a = b);
                    this.lw = a;
                    b = this.Zs;
                    this.Zs = !(!c || !(Math.abs(a.Sh) > c.Sh / 2 + 512 || Math.abs(a.Th) > c.Th / 2 + 512));
                    b && this.Zs || (this.uw && this.map && (c = _.Ka(this.map), (c = cS.get(c)) && ZWa(c, this)), (new _.Jo(a.Sh, a.Th)).equals(this.Rk) || (sS(this, new _.Jo(a.Sh, a.Th)), this.Kz(this.vC)), this.vC = !1, this.sA = this.tA = !0)
                } else this.Zs = !0, this.lw = null, sS(this, null)
            }
            Kz(a) {
                this.Pl = null;
                this.Hj && this.Hj.nh && this.Hj.Kz(this.Uv());
                lS(this, a)
            }
            Ym(a) {
                this.LD !== a && (this.LD = a, this.kn.set("map", this.LD ? this.map : null))
            }
            yy() {
                if (!aS(this) ||
                    this.Wo || !pS(this).length) return null;
                var a = this.map.getProjection();
                if (!a) return null;
                a = a.fromLatLngToPoint(this.Sp);
                const b = [];
                for (const f of pS(this)) {
                    a: {
                        var c = f;
                        var d = this.Rk;
                        var e = this.Wx;
                        if (!d) {
                            d = {
                                size: new _.No(0, 0),
                                offset: new _.Jo(0, 0)
                            };
                            break a
                        }
                        const {
                            size: k,
                            offset: n
                        } = JVa(this, c);c = ER(e);d = {
                            size: k,
                            offset: new _.Jo(c.offsetX - d.x + n.x, c.offsetY - d.y + n.y)
                        }
                    }
                    const {
                        size: g,
                        offset: h
                    } = d;d = new DXa(a.x, a.y, g.width, g.height, h.x, h.y);b.push(d)
                }
                return b
            }
            zt() {}
            xq() {
                return this
            }
            LF(a) {
                return !this.position || this.Oz ?
                    !1 : HVa(a, this)
            }
            Zw() {
                const a = this.xq();
                this.zn ? a.setAttribute("role", "button") : this.title ? a.setAttribute("role", "img") : a.removeAttribute("role")
            }
            get Nl() {
                return this.Hj ? this.Hj.Nl() : !1
            }
            Um() {
                sS(this, null);
                jS(this);
                this.tA && this.Mh && this.Kk && (this.Mh.Ql(this.Kk), this.Kk = null);
                this.remove();
                this.uw = !0
            }
            dispose() {
                if (!this.EB) {
                    this.EB = !0;
                    try {
                        this.Mk && (qS(this), this.Um())
                    } finally {
                        this.EB = !1
                    }
                }
            }
            TC(a) {
                {
                    const c = this.ui ? .get("projectionController");
                    if (this.ui && a && c) {
                        var b = this.ui.uq.getBoundingClientRect();
                        a =
                            c.fromContainerPixelToLatLng(new _.Jo(a.clientX - b.left, a.clientY - b.top))
                    } else a = null
                }
                a && (this.position = a)
            }
            ho() {
                var a = this.ui ? .get("projectionController");
                if (!this.ui || !a || !this.Sp) return null;
                a = a.fromLatLngToContainerPixel(this.Sp);
                const b = this.ui.uq.getBoundingClientRect();
                return {
                    clientX: a.x + b.left,
                    clientY: a.y + b.top
                }
            }
            connectedCallback() {
                super.connectedCallback();
                this.vk.appendChild(this.Gj);
                this.Hj && iS(this.Hj);
                if (this.isConnected && this.parentNode) {
                    const a = BR(this);
                    a ? this.setMap(a) : (qS(this), console.error("AdvancedMarkerElement: parent element must be a <gmp-map>."))
                }
            }
            disconnectedCallback() {
                !this.isConnected &&
                    this.sA && (this.map = null);
                this.uw = !0;
                super.disconnectedCallback()
            }
        };
    xS.prototype.addListener = xS.prototype.addListener;
    xS.prototype.addEventListener = xS.prototype.addEventListener;
    xS.prototype.constructor = xS.prototype.constructor;
    xS.Gi = {
        Ki: 181577,
        Ji: 181576
    };
    _.A([_.fr({
        Ih: "anchor-top",
        type: String,
        Lh: !0
    }), _.B("design:type", Object), _.B("design:paramtypes", [Object])], xS.prototype, "anchorTop", null);
    _.A([_.fr({
        Ih: "anchor-left",
        type: String,
        Lh: !0
    }), _.B("design:type", Object), _.B("design:paramtypes", [Object])], xS.prototype, "anchorLeft", null);
    _.A([_.fr({
        Ih: "gmp-clickable",
        type: Boolean,
        Lh: !0
    }), _.B("design:type", Object), _.B("design:paramtypes", [Object])], xS.prototype, "gmpClickable", null);
    _.A([_.fr({
        li: _.tt,
        ij: _.Jp,
        Lh: !0
    }), _.B("design:type", Object), _.B("design:paramtypes", [Object])], xS.prototype, "position", null);
    _.A([_.fr({
        li: {
            zk: a => a || "",
            bk: a => a || null
        },
        Lh: !0
    }), _.B("design:type", String), _.B("design:paramtypes", [String])], xS.prototype, "title", null);
    var NXa = !1,
        OXa = class extends xS {
            constructor(a = {}) {
                super(a);
                if (!wS) throw Error("InternalUseAdvancedMarkerView is not allowed to be instantiated directly.");
            }
        };
    _.qp("gmp-internal-use-am", OXa);
    var yS = {
        Marker: _.lt,
        CollisionBehavior: _.ht,
        Animation: _.Eha,
        PJ: () => {},
        pA: function(a, b, c) {
            const d = _.vGa();
            if (b instanceof _.$o) BWa(a, b, d);
            else {
                const e = new _.jr;
                BWa(e, b, d);
                const f = new _.jr;
                c || SWa(f, b, d);
                new HXa(a, f, e, c)
            }
        },
        CE: function(a = {}) {
            wS = !0;
            a = new OXa(a);
            wS = !1;
            return a
        },
        AdvancedMarkerElement: xS,
        PinElement: _.iC,
        AdvancedMarkerClickEvent: oS,
        AdvancedMarkerView: void 0,
        PinView: void 0,
        connectForExplicitThirdPartyLoad: () => {
            const a = {
                AdvancedMarkerElement: xS,
                PinElement: _.iC,
                AdvancedMarkerClickEvent: oS,
                AdvancedMarkerView: void 0,
                PinView: void 0
            };
            _.Hm(a);
            _.za.google.maps.marker = a;
            NXa || (NXa = !0, _.qp("gmp-advanced-marker", xS))
        }
    };
    _.Im(yS, ["PJ", "pA", "CE", "connectForExplicitThirdPartyLoad"]);
    _.Hm(yS);
    _.Sl("marker", yS);
});