google.maps.__gjsload__('poly', function(_) {
    var YS = function(a, b) {
            return b ? _.Cx(b, a.map.get("projection")) : null
        },
        ZS = function(a, b) {
            return b ? _.Ar(b, a.map.get("projection"), !1) : null
        },
        $S = function(a) {
            for (a = a.toString(16); a.length < 2;) a = "0" + a;
            return a
        },
        JZa = function(a) {
            const b = a.poly;
            if (b) {
                var c = a.get("radius"),
                    d = a.get("center");
                a = a.get("planetRadius");
                _.vm(c) && d && a ? (c /= Number(a), c = _.WFa(new _.Ep(d), c).map(e => e.map(f => new _.pn(f, !0))), b.setPaths(c)) : b.setPaths([])
            }
        },
        KZa = function(a, b, c, d) {
            let e = null,
                f = c * c;
            for (let v = 0, w = d.length; v < w; ++v) {
                a: {
                    let y, E;
                    var g =
                        void 0;
                    var h = a;
                    var k = b,
                        n = c,
                        p = d[v];
                    const H = p.length;
                    if (!H || n <= 0) {
                        h = null;
                        break a
                    }
                    let M = 0,
                        I = n * n * 1.01;
                    const F = [0, 0];
                    let W = p[M++] - h,
                        qa = p[M++] - k,
                        ta = (W < -n ? 1 : 0) | (W > n ? 2 : 0) | (qa < -n ? 4 : 0) | (qa > n ? 8 : 0);
                    var r = W * W + qa * qa;!ta && r <= I && (I = r, F[0] = W, F[1] = qa);
                    for (; M < H;) {
                        E = W;
                        y = qa;
                        g = ta;
                        W = p[M++] - h;
                        qa = p[M++] - k;
                        ta = (W < -n ? 1 : 0) | (W > n ? 2 : 0) | (qa < -n ? 4 : 0) | (qa > n ? 8 : 0);
                        if (g & ta) continue;
                        r = W * W + qa * qa;
                        !ta && r <= I && (I = r, F[0] = W, F[1] = qa);
                        g = W - E;
                        const ya = qa - y;
                        var t = g * g + ya * ya;
                        if (t < 1E-12) continue;
                        const Fa = W * g + qa * ya;
                        if (Fa < 0 || Fa > t) continue;
                        r -= Fa * Fa / t;
                        r <=
                            I && (I = r, t = 1 - Fa / t, F[0] = E + g * t, F[1] = y + ya * t)
                    }
                    F[0] += h;F[1] += k;F[2] = I;h = I <= n * n ? F : null
                }
                h && h[2] <= f && (f = h[2], e = h)
            }
            return e
        },
        LZa = function(a, b, c) {
            let d = 0;
            for (let e = 0, f = c.length; e < f; ++e) d += _.cGa(a, b, c[e]);
            return !!(d & 1)
        },
        aT = function(a, b, c) {
            const d = a.path;
            a = a.ao;
            const e = [],
                f = a ? [] : null,
                g = d[0] === d[d.length - 2] && d[1] === d[d.length - 1];
            for (let h = 0, k = d.length - 2; h < k; h += 2) {
                const n = d[h],
                    p = d[h + 1],
                    r = d[h + 2],
                    t = d[h + 3];
                let v = null,
                    w = null;
                a && (v = a[h / 2], w = a[h / 2 + 1]);
                let y, E;
                switch (b) {
                    case 0:
                        y = n >= c;
                        E = r >= c;
                        break;
                    case 1:
                        y = n <= c;
                        E = r <= c;
                        break;
                    case 2:
                        y = p >= c;
                        E = t >= c;
                        break;
                    case 3:
                        y = p <= c, E = t <= c
                }!h && y && (e.push(n, p), a && f.push(v));
                if (y !== E) {
                    let H = 0;
                    switch (b) {
                        case 0:
                        case 1:
                            H = (c - n) / (r - n);
                            e.push(c, p + H * (t - p));
                            break;
                        case 2:
                        case 3:
                            H = (c - p) / (t - p), e.push(n + H * (r - n), c)
                    }
                    a && f.push(v + H * (w - v))
                }
                E && (e.push(r, t), a && f.push(w))
            }
            g && e.length && (e[0] !== e[e.length - 2] || e[1] !== e[e.length - 1]) && (e.push(e[0], e[1]), a && f.push(a[a.length - 1]));
            return {
                path: e,
                ao: f
            }
        },
        MZa = function(a, b) {
            a = aT(a, 0, b.minX);
            a = aT(a, 1, b.maxX);
            a = aT(a, 2, b.minY);
            return aT(a, 3, b.maxY)
        },
        NZa = function(a) {
            const b =
                a.length;
            !b || a[0] === a[b - 2] && a[1] === a[b - 1] || (a.push(a[0]), a.push(a[1]))
        },
        OZa = function(a, b) {
            for (let d = 0, e = a.length; d < e; d += 2) {
                var c = new _.pn(a[d], a[d + 1], !0);
                c = _.Cx(c, b);
                a[d] = c.mh;
                a[d + 1] = c.nh
            }
        },
        PZa = function(a) {
            const b = new _.FHa(["mousemove", "mouseup", "mousewheel"]);
            _.$I(b, ["panes", "pixelBounds"], a);
            return b
        },
        bT = function(a) {
            const b = PZa(a);
            _.Hn(b, "mousemove", _.Cn);
            _.Hn(b, "mouseup", _.Cn);
            _.Hn(b, "mousewheel", (...c) => {
                const d = a.get("mouseEventTarget");
                d && _.Wn(d, "mousewheel", ...c)
            });
            return b
        },
        RZa = function(a) {
            var b = ["px", "%"];
            if (!a || !_.pm(b)) return null;
            if (a === "0") return {
                value: 0,
                unit: b[0]
            };
            for (let c = 0, d = b.length; c < d; ++c) {
                const e = b[c],
                    f = (QZa[e] = QZa[e] || new RegExp(`^(\\d+(?:\\.\\d+)?)${e}$`)).exec(a);
                if (f) return {
                    value: Number(f[1]),
                    unit: e
                }
            }
            return null
        },
        SZa = function(a, b, c) {
            const d = _.Ar(new _.Oq(c.x, c.y), a);
            a = _.Ar(new _.Oq(c.x + b, c.y + b), a);
            return Math.min(Math.abs(d.lat() - a.lat()), Math.abs(d.lng() - a.lng()))
        },
        cT = function(a, b) {
            return a.x * b.x + a.y * b.y + a.z * b.z
        },
        dT = function(a, b, c) {
            c.x = a.y * b.z - a.z * b.y;
            c.y = a.z * b.x - a.x * b.z;
            c.z = a.x * b.y - a.y * b.x
        },
        eT = function(a, b) {
            const c = _.vl(a[0]);
            a = _.vl(a[1]);
            const d = Math.cos(c);
            b.x = Math.cos(a) * d;
            b.y = Math.sin(a) * d;
            b.z = Math.sin(c)
        },
        fT = function(a, b) {
            const c = Math.atan2(a.y, a.x);
            b[0] = _.wl(Math.atan2(a.z, Math.sqrt(a.x * a.x + a.y * a.y)));
            b[1] = _.wl(c)
        },
        UZa = function(a, b, c) {
            c.x = a.x + b.x;
            c.y = a.y + b.y;
            c.z = a.z + b.z;
            TZa(c)
        },
        TZa = function(a) {
            const b = Math.sqrt(cT(a, a));
            b < 1E-12 || (a.x /= b, a.y /= b, a.z /= b)
        },
        VZa = function(a, b) {
            for (; b < a.mh.length; ++b) a.mh[b].index = b
        },
        gT = function(a, b, c, d) {
            const e = b.index,
                f = b.Cp.get("position");
            b = (b = b.Cp.get("latLngPosition")) && a.map ? _.Cx(b, a.map.get("projection")) : null;
            _.Wn(a, c, e, f, b, d)
        },
        WZa = function(a, b, c) {
            if ((b = b.Cp.Hh) && a.map) {
                a = a.map.__gm.get("projectionController");
                const d = a.fromLatLngToDivPixel(c);
                b = b(d);
                if (b = a.fromDivPixelToLatLng(b)) return b
            }
            return c
        },
        XZa = function(a, b) {
            if (b && a.rq) {
                if (a.constraint === 2) return {
                    clientX: a.rq.clientX,
                    clientY: b.clientY
                };
                if (a.constraint === 1) return {
                    clientX: b.clientX,
                    clientY: a.rq.clientY
                }
            }
            return b
        },
        hT = function(a) {
            const b = a.nh,
                c = a.get("center");
            var d = a.get("radius");
            if (c && _.vm(d)) {
                a.ph = !0;
                var e = a.get("planetRadius"),
                    f = _.wl(d / e);
                var g = _.vl(c.lat());
                if (e = d / e) {
                    d = Math.cos(e);
                    e = Math.sin(e);
                    var h = Math.sin(g);
                    g = Math.atan2(Math.sin(Math.acos((1 - d) / e * Math.tan(g))) * e * Math.cos(g), d - h * h)
                } else g = 0;
                g = _.wl(g);
                b.setAt(0, c);
                d = c.lat() + f;
                d < 90 ? b.setAt(1, new _.pn(d, c.lng())) : b.setAt(1, null);
                f = c.lat() - f;
                f > -90 ? b.setAt(2, new _.pn(f, c.lng())) : b.setAt(2, null);
                g <= 180 ? (f = c.lng() + g, g = c.lng() - g, b.setAt(3, new _.pn(c.lat(), f)), b.setAt(4, new _.pn(c.lat(), g))) : (b.setAt(3, null), b.setAt(4, null));
                a.ph = !1;
                a.oh || _.Wn(a, "toolbar", {
                    show: !1
                })
            } else b.clear()
        },
        YZa = function(a) {
            const b = a.qh;
            b && (b.unbindAll(), b.release(), _.Mn(b), a.qh = null)
        },
        a_a = function(a) {
            var b = a.get("panes");
            if (!b) return null;
            b = new iT(a.nh, ZZa, $Za, !1, b.overlayMouseTarget, a.rh, a.get("map"), a.Mh);
            b.bindTo("projection", a);
            b.bindTo("zoom", a);
            b.bindTo("projectionCenterQ", a);
            b.bindTo("panningEnabled", a);
            b.bindTo("mapPixelBounds", a);
            b.bindTo("color", a);
            b.bindTo("zIndex", a);
            b.bindTo("offset", a);
            a.mh.bindTo("freeVertexPosition", b);
            let c = null,
                d = null;
            _.Hn(b, "dragstart", e => {
                c = a.get("center");
                d = a.get("radius");
                a.mh.set("freeVertexIsCenter", e === 0);
                a.mh.set("map", a.get("map"))
            });
            _.Hn(b, "dragend", (e, f, g) => {
                a.mh.set("map", null);
                _.Wn(a, "toolbar", {
                    show: !0,
                    Ex: g,
                    action: () => {
                        a.set("center", c);
                        a.set("radius", d)
                    }
                })
            });
            _.Vn(b, "dragstart", a);
            _.Vn(b, "dragend", a);
            _.Vn(b, "panbynow", a);
            return b
        },
        b_a = function(a, b, c) {
            return c && b || a
        },
        jT = function(a) {
            let b;
            if (a.nh) {
                const c = _.XFa(a.get("color"));
                c && (c.red = Math.floor((c.red + 255) / 2), c.green = Math.floor((c.green + 255) /
                    2), c.blue = Math.floor((c.blue + 255) / 2), b = ["#", $S(c.red), $S(c.green), $S(c.blue)].join(""))
            }
            a.uh.setFillColor(a.mh, b || "white")
        },
        e_a = function(a, b, c, d) {
            function e() {
                a.get("editable") ? c_a(a, b, c, d) : (d_a(a), _.Wn(a, "toolbar", {
                    show: !1
                }))
            }
            a.editable_changed = e;
            e()
        },
        c_a = function(a, b, c, d) {
            if (!a.oh) {
                var e = new _.oN(a, !0);
                a.qh = e;
                var f = new f_a;
                f.bindTo("strokeColor", e);
                f.bindTo("strokeOpacity", e, "ghostStrokeOpacity");
                f.bindTo("strokeWeight", e);
                f.bindTo("center", a);
                f.bindTo("radius", a);
                f.bindTo("planetRadius", c);
                f.bindTo("zIndex",
                    a);
                a.ph = f;
                var g = b.__gm;
                a.mh = bT(g);
                var h = _.vM(g, a),
                    k = _.wx() ? 9 : 0,
                    n = new kT;
                d = new g_a(f, function(p, r, t) {
                    return new lT(p, a.mh, k, r, h, n, t)
                }, d);
                d.set("map", b);
                d.bindTo("center", a);
                d.bindTo("radius", a);
                d.bindTo("planetRadius", c);
                d.bindTo("panes", g);
                d.bindTo("projection", b);
                d.bindTo("zoom", g);
                d.bindTo("projectionCenterQ", g);
                d.bindTo("panningEnabled", b, "draggable");
                d.bindTo("mapPixelBounds", g, "pixelBounds");
                d.bindTo("offset", g);
                d.bindTo("color", e, "strokeColor");
                d.bindTo("zIndex", a);
                a.oh = d;
                _.Vn(d, "panbynow",
                    g);
                _.Vn(d, "toolbar", a)
            }
        },
        d_a = function(a) {
            const b = a.oh;
            b && (b.unbindAll(), b.set("map", null), _.Mn(b), delete a.oh, a.mh ? .unbindAll(), a.mh ? .release(), delete a.mh, a.ph ? .unbindAll(), delete a.ph, a.qh ? .release(), delete a.qh)
        },
        mT = function(a, b) {
            const c = a.__gm,
                d = () => {
                    const e = a.get("gestureHandling");
                    e != null ? b.set("panningEnabled", e !== "none") : b.set("panningEnabled", a.get("draggable"))
                };
            _.Hn(a, "gesturehandling_changed", d);
            _.Hn(a, "draggable_changed", d);
            b.bindTo("panes", c);
            b.bindTo("projectionController", c);
            b.bindTo("containerPixelBounds",
                c, "pixelBounds");
            c.oi = _.Vn(b, "panbynow", c)
        },
        nT = function(a, b) {
            const c = () => {
                const d = a.getMap();
                a.getDraggable() && _.N(d, 147751)
            };
            a.draggable_changed = c;
            c();
            a.wh = [_.Hn(a, "mouseover", d => {
                d.vertex == null && d.edge == null && b.set("poly", a)
            }), _.Hn(a, "mouseout", d => {
                d.vertex != null || d.edge != null || d.domEvent && _.iv(d.domEvent) || b.set("poly", null)
            })]
        },
        oT = function(a) {
            (a.wh || []).forEach(_.Jn);
            delete a.draggable_changed;
            delete a.wh
        },
        pT = function(a) {
            if (!a.get("active")) {
                var b = a.get("panes"),
                    c = a.get("projectionController"),
                    d = a.get("poly");
                if (b !== a.th || c !== a.ph || d !== a.mh) a.th = b, a.ph = c, a.mh = d, a.nh && a.nh.forEach(_.Jn), a.nh = null, a.th && a.ph && a.mh ? (a.nh = [_.Rn(a.mh, "mousedown", a, a.Bh), _.Rn(a.mh, "mouseup", a, a.Ch), _.Rn(a.mh, "movestart", a, a.yh), _.Rn(a.mh, "move", a, a.zh), _.Rn(a.mh, "moveend", a, a.xh)], a.nh.push(_.Hn(a.mh, "editable_changed", () => {
                    a.get("storeEditable") && (a.uh = a.mh.get("editable"), a.set("storeEditable", !1), a.mh.set("editable", !1), a.set("storeEditable", !0))
                })), a.bindTo("draggable", a.mh)) : (a.unbind("draggable"), a.set("draggable", !1))
            }
        },
        qT = function(a, b, c) {
            var d = a.get("position");
            d = a.ph.fromDivPixelToLatLng(d);
            c = new _.MB(d, c.domEvent);
            a.mh && _.Wn(a.mh, b, c)
        },
        h_a = function(a) {
            _.DM(a.qh, a.get("panningEnabled") !== !1 && a.get("dragging"))
        },
        k_a = async function(a, b) {
            const c = await _.Rl("geometry");
            a.mh = new rT((d, e, f) => new i_a(d, e, f, c.spherical.computeHeading, c.spherical.computeOffsetOrigin), a.map, b);
            mT(a.map, a.mh);
            a.ui.xh.forEach(d => {
                j_a(a, d)
            })
        },
        j_a = function(a, b) {
            if (a.mh && a.Mh) {
                var c = new _.PM(["baseMapType"], "planetRadius", _.Nba);
                c.bindTo("baseMapType",
                    a.ui);
                var d = b.rh = new l_a(b);
                d.set("map", a.map);
                d.bindTo("radius", b);
                d.bindTo("center", b);
                d.bindTo("capturing", b);
                d.bindTo("clickable", b);
                d.bindTo("cursor", b);
                d.bindTo("fillColor", b);
                d.bindTo("fillOpacity", b);
                d.bindTo("strokeColor", b);
                d.bindTo("strokeOpacity", b);
                d.bindTo("strokeWeight", b);
                d.bindTo("strokePosition", b);
                d.bindTo("zIndex", b);
                d.bindTo("suppressUndo", b);
                d.bindTo("planetRadius", c);
                var e = [];
                sT.forEach(f => {
                    e.push(_.Vn(d, f, b))
                });
                e.push(_.Vn(b, "toolbar", d));
                b.Wh = e;
                nT(b, a.mh);
                e_a(b, a.map, c, a.Mh);
                _.N(a.map, 147750)
            }
        },
        n_a = function(a) {
            const b = a.getVisible() != 0 ? a.getMap() : null;
            a.nh !== b && (a.nh && a.nh.__gm.xh.remove(a), b && (b.__gm.xh.mh || new m_a(b), _.rq(b.__gm.xh, a)), a.nh = b)
        },
        o_a = function(a) {
            a = a.poly.get("latLngs");
            const b = new tT(0),
                c = new tT(0);
            a.forEach(d => {
                d.forEach(e => {
                    eT([e.lat(), e.lng()], c);
                    b.x += c.x;
                    b.y += c.y;
                    b.z += c.z
                })
            });
            TZa(b);
            a = [0, 0];
            fT(b, a);
            a[0] = _.sm(a[0], -89, 89);
            return new _.pn(a[0], a[1])
        },
        q_a = function(a, b, c, d) {
            a.capturing_changed = () => {
                if (a.get("capturing"))
                    if (a.get("clickable") == 0) {
                        const e =
                            a.oh = new p_a;
                        e.bindTo("draggableCursor", a, "cursor");
                        e.set("active", !0);
                        b.bindTo("cursor", e);
                        d.register(e)
                    } else d.setCapture(c, a);
                else a.oh ? (d.unregister(a.oh), a.oh.unbindAll(), b.unbind("cursor"), b.set("cursor", ""), delete a.oh) : d.releaseCapture(c, a)
            }
        },
        r_a = function(a) {
            a.nh || (a.nh = b => {
                _.qm(a.nh.Wk, (c, d) => {
                    d(b)
                })
            }, a.nh.Wk = {});
            a.ph || (a.ph = (b, c) => {
                _.qm(a.ph.Wk, (d, e) => {
                    e(b, c)
                })
            }, a.ph.Wk = {});
            a.oh || (a.oh = (b, c) => {
                _.qm(a.oh.Wk, (d, e) => {
                    e(b, c)
                })
            }, a.oh.Wk = {})
        },
        uT = function(a, b) {
            a.nh && delete a.nh.Wk[b];
            a.ph && delete a.ph.Wk[b];
            a.oh && delete a.oh.Wk[b]
        },
        v_a = function(a, b, c) {
            const d = _.Zn(b);
            c in s_a && (a.mh[d] = b, vT(b));
            c in t_a && (a.nh[d] = b);
            u_a(a)
        },
        vT = function(a) {
            var b = a.get("latLngs");
            if (b) {
                a = _.Zn(a);
                uT(b, a);
                b = b.getArray();
                for (let c = 0, d = b.length; c < d; ++c) uT(b[c], a)
            }
        },
        u_a = function(a) {
            a.timeout || (a.timeout = _.XI(() => {
                a.timeout = 0;
                var b = a.nh;
                a.nh = {};
                const c = a.mh;
                a.mh = {};
                for (var d of Object.keys(b)) {
                    var e = b[d],
                        f = e.ph;
                    f.style = w_a(a.ph, e);
                    f.kC && !c[d] && f.kC()
                }
                b = a.oh;
                d = a.get("projection");
                for (const g of Object.keys(c)) e = c[g], f = e.ph, b.remove(f),
                    f.geometry = x_a(e, d), _.rq(b, f), y_a(a, e)
            }))
        },
        y_a = function(a, b) {
            function c() {
                v_a(e, b, "latLngs")
            }
            const d = b.get("latLngs");
            if (d) {
                var e = a,
                    f = _.Zn(b);
                a = d.getArray();
                for (let g = 0, h = a.length; g < h; ++g) {
                    const k = a[g];
                    r_a(k);
                    k.nh.Wk[f] = k.ph.Wk[f] = k.oh.Wk[f] = c
                }
                r_a(d);
                d.ph.Wk[f] = d.oh.Wk[f] = (g, h) => {
                    uT(h, f);
                    c()
                };
                d.nh.Wk[f] = c
            }
        },
        z_a = function(a) {
            const b = a.get("projection"),
                c = a.nh;
            if (b) {
                var d = a.mh,
                    e = d.getLength(),
                    f = a.get("geodesic");
                for (let g = 0; g < e - 1; ++g) c.setAt(g, wT(d.getAt(g), d.getAt(g + 1), f, b, a.interpolate))
            } else c.clear()
        },
        A_a = function(a, b, c) {
            const d = a.get("projection");
            if (d) {
                var e = a.mh,
                    f = e.getAt(c),
                    g = a.nh,
                    h = a.get("geodesic");
                c > 0 && g.setAt(c - 1, wT(e.getAt(c - 1), f, h, d, a.interpolate));
                c < e.getLength() - 1 && (a = wT(f, e.getAt(c + 1), h, d, a.interpolate), b ? g.insertAt(c, a) : g.setAt(c, a))
            }
        },
        wT = function(a, b, c, d, e) {
            c ? d = e(a, b, .5) : (Math.abs(a.lng() - b.lng()) > 180 && (a = new _.pn(a.lat(), _.tm(a.lng(), b.lng() - 180, b.lng() + 180), !0)), a = d.fromLatLngToPoint(a), b = d.fromLatLngToPoint(b), d = d.fromPointToLatLng(new _.Jo((a.x + b.x) / 2, (a.y + b.y) / 2)));
            return d
        },
        xT = function(a) {
            a.unbindAll();
            a.release();
            _.Mn(a);
            a.nh && (a.nh.release(), a.nh.unbindAll())
        },
        AT = function(a) {
            function b() {
                _.Wn(c, "toolbar", {
                    show: !1
                })
            }
            const c = a;
            B_a(c);
            const d = c.get("paths");
            a = c.get("panes");
            if (d && a) {
                var e = a.overlayMouseTarget;
                d.forEach(f => {
                    c.nh.push(yT(c, f, e));
                    c.get("suppressGhostControlPoints") || c.mh.push(zT(c, f, e))
                });
                b();
                a = c.oh;
                a.push(_.Hn(d, "insert_at", f => {
                    const g = c.get("suppressGhostControlPoints"),
                        h = d.getAt(f);
                    c.nh.splice(f, 0, yT(c, h, e));
                    g || c.mh.splice(f, 0, zT(c, h, e));
                    b()
                }));
                a.push(_.Hn(d,
                    "remove_at", f => {
                        const g = c.get("suppressGhostControlPoints");
                        xT(c.nh[f]);
                        c.nh.splice(f, 1);
                        g || (xT(c.mh[f]), c.mh.splice(f, 1));
                        b()
                    }));
                a.push(_.Hn(d, "set_at", f => {
                    const g = c.get("suppressGhostControlPoints");
                    xT(c.nh[f]);
                    const h = d.getAt(f);
                    c.nh[f] = yT(c, h, e);
                    g || (xT(c.mh[f]), c.mh[f] = zT(c, h, e));
                    b()
                }))
            }
        },
        B_a = function(a) {
            a.nh.forEach(xT);
            a.mh.forEach(xT);
            a.nh.length = 0;
            a.mh.length = 0;
            a.oh.forEach(_.Jn);
            a.oh.length = 0
        },
        D_a = function(a, b, c, d) {
            const e = new iT(b, ["pointer"], [0], d, c, a.ph, a.get("map"), a.Mh);
            e.bindTo("projection",
                a);
            e.bindTo("zoom", a);
            e.bindTo("projectionCenterQ", a);
            e.bindTo("panningEnabled", a);
            e.bindTo("mapPixelBounds", a);
            e.bindTo("color", a);
            e.bindTo("zIndex", a);
            e.bindTo("offset", a);
            _.Vn(e, "dragstart", a);
            _.Vn(e, "dragend", a);
            _.Vn(e, "panbynow", a);
            _.Vn(e, "toolbar", a);
            sT.forEach(f => {
                _.Hn(e, f, (g, h, k, n) => {
                    h = b.getAt(g);
                    let p;
                    if (a.Gn) {
                        k = d ? a.mh : a.nh;
                        for (let r = 0; r < k.length; ++r)
                            if (k[r] === e) {
                                p = r;
                                break
                            }
                    }
                    _.Wn(a, f, new C_a(h, n, p, d ? void 0 : g, d ? g : void 0))
                })
            });
            return e
        },
        yT = function(a, b, c) {
            const d = D_a(a, b, c, !1),
                e = a.Qm;
            let f;
            _.Hn(d, "dragstart", g => {
                f = b.getAt(g);
                var h = b.getArray(),
                    k = a.Gn;
                const n = h.length;
                if (n < 2) e.set("anchors", []);
                else {
                    var p = h[k && g === 0 ? n - 1 : g - 1];
                    g = h[k && g === n - 1 ? 0 : g + 1];
                    h = [];
                    p && h.push(p);
                    g && h.push(g);
                    e.set("anchors", h)
                }
                e.bindTo("freeVertexPosition", d);
                e.set("map", a.get("map"))
            });
            _.Hn(d, "dragend", (g, h, k) => {
                e.set("map", null);
                _.Wn(a, "toolbar", {
                    show: !0,
                    Ex: k,
                    action: () => {
                        b.setAt(g, f)
                    }
                })
            });
            return d
        },
        zT = function(a, b, c) {
            const d = new _.xp,
                e = a.Gn ? new E_a(b) : b,
                f = new F_a(e, d, a.interpolate);
            f.bindTo("geodesic", a);
            f.bindTo("projection",
                a);
            const g = D_a(a, d, c, !0);
            g.nh = f;
            const h = a.Qm;
            _.Hn(g, "dragstart", k => {
                k = [e.getAt(k), e.getAt(k + 1)];
                h.set("anchors", k);
                h.bindTo("freeVertexPosition", g);
                h.set("map", a.get("map"))
            });
            _.Hn(g, "dragend", (k, n, p) => {
                b.insertAt(k + 1, d.getAt(k));
                h.set("map", null);
                _.Wn(a, "toolbar", {
                    show: !0,
                    Ex: p,
                    action: () => {
                        b.removeAt(k + 1)
                    }
                })
            });
            return g
        },
        I_a = function(a, b, c, d, e) {
            function f() {
                a.get("editable") ? (G_a(a, b, d, e), _.N(b, 147756)) : (_.Wn(c, "toolbar", {
                    show: !1,
                    poly: a
                }), H_a(a))
            }
            a.editable_changed = f;
            f();
            a.th = _.Hn(a, "toolbar", g => {
                g.poly =
                    a;
                _.Wn(c, "toolbar", g)
            })
        },
        G_a = function(a, b, c, d) {
            if (!a.sh) {
                var e = new _.oN(a, a.Gn);
                a.uh = e;
                var f = new _.GHa;
                f.bindTo("strokeColor", e);
                f.bindTo("strokeOpacity", e, "ghostStrokeOpacity");
                f.bindTo("strokeWeight", e);
                f.bindTo("geodesic", a);
                f.bindTo("zIndex", a);
                a.Qm = f;
                var g = b.__gm;
                a.nh = bT(g);
                var h = _.vM(g, a),
                    k = _.wx() ? 9 : 0,
                    n = new kT,
                    p = new J_a(f, a.Gn, function(r, t, v) {
                        return new lT(r, a.nh, k, t, h, n, v)
                    }, c, d);
                p.set("map", b);
                p.bindTo("paths", a, "latLngs");
                p.bindTo("panes", g);
                p.bindTo("projection", b);
                p.bindTo("zoom", g);
                p.bindTo("projectionCenterQ",
                    g);
                p.bindTo("panningEnabled", b, "draggable");
                p.bindTo("mapPixelBounds", g, "pixelBounds");
                p.bindTo("offset", g);
                p.bindTo("color", e, "strokeColor");
                p.bindTo("zIndex", a);
                p.bindTo("geodesic", a);
                p.bindTo("suppressGhostControlPoints", a);
                a.sh = p;
                _.Vn(p, "panbynow", g);
                _.Vn(p, "toolbar", a);
                sT.forEach(r => {
                    _.Vn(p, r, a)
                })
            }
        },
        H_a = function(a) {
            const b = a.sh;
            b && (b.unbindAll(), b.set("map", null), _.Mn(b), delete a.sh, a.nh.unbindAll(), a.nh.release(), delete a.nh, a.Qm.unbindAll(), delete a.Qm, a.uh.release(), delete a.uh)
        },
        K_a = function(a) {
            a.nh &&
                _.Ln(a.nh, "suppressundo_changed")
        },
        L_a = function(a, b, c) {
            a.oh(_.cp);
            a.ph = b;
            K_a(a);
            a.nh = c;
            _.cJ(a.mh);
            _.Hn(c, "suppressundo_changed", () => {
                c.get("suppressUndo") && a.Pk()
            })
        },
        M_a = function(a) {
            return (a || []).map(b => (b || []).map(c => Math.round(c)))
        },
        P_a = function(a, b, c) {
            function d() {
                g[v++] = k.latLng[0];
                g[v++] = k.latLng[1];
                f && (h[r++] = k.distance);
                n = k
            }
            const e = a.path,
                f = a.ao;
            if (!e.length) return a;
            const g = Array(e.length),
                h = f ? Array(f.length) : null;
            a = [];
            let k, n = BT();
            g[0] = n.latLng[0] = e[0];
            g[1] = n.latLng[1] = e[1];
            f && (h[0] = n.distance =
                f[0]);
            n.depth = 0;
            eT(n.latLng, n.Fu);
            let p = 1,
                r = 1;
            const t = [];
            let v = 2;
            for (let y = 2; y < e.length || t.length;) {
                t.length ? k = t.pop() : (k = BT(), k.depth = 0, k.latLng[0] = e[y++], k.latLng[1] = e[y++], f && (k.distance = f[p++]), eT(k.latLng, k.Fu));
                if (Math.max(n.depth, k.depth) >= 12) {
                    d();
                    continue
                }
                var w = new _.rp;
                w.minX = Math.min(n.latLng[0], k.latLng[0]);
                w.maxX = Math.max(n.latLng[0], k.latLng[0]);
                w.minY = Math.min(n.latLng[1], k.latLng[1]);
                w.maxY = Math.max(n.latLng[1], k.latLng[1]);
                if (!_.tp(b, w)) {
                    d();
                    continue
                }
                const E = BT();
                UZa(n.Fu, k.Fu, E.Fu);
                fT(E.Fu, E.latLng);
                E.depth = Math.max(n.depth, k.depth) + 1;
                f && (E.distance = (n.distance + k.distance) / 2);
                const H = w.minY - 1E-6,
                    M = w.maxY + 1E-6;
                w = E.latLng;
                w[1] = N_a(H, M, w[1]);
                O_a(n.latLng, k.latLng, a);
                Math.max(Math.abs(E.latLng[0] - a[0]), Math.abs(E.latLng[1] - a[1])) <= c ? d() : (t.push(k), t.push(E))
            }
            return {
                path: g,
                ao: h
            }
        },
        O_a = function(a, b, c) {
            c[0] = (a[0] + b[0]) / 2;
            c[1] = (a[1] + b[1]) / 2
        },
        N_a = function(a, b, c) {
            for (; c < a;) c += 360;
            for (; c > b;) c -= 360;
            return c
        },
        BT = function() {
            return {
                latLng: [0, 0],
                Fu: new tT(0)
            }
        },
        Q_a = function(a) {
            return function(b,
                c) {
                if (b.unit === "px") b = b.value / a;
                else {
                    const d = _.pm(c);
                    b = d ? b.value / 100 * c[d - 1] : 0
                }
                return b
            }
        },
        R_a = function(a, b) {
            return function(c, d) {
                if (c === 0) return !0;
                c = a[c];
                d = a[d];
                for (let e = 0, f; f = b[e]; ++e) {
                    const g = f.offset,
                        h = f.repeat;
                    if (h) {
                        if (Math.floor((c - g) / h) !== Math.floor((d - g) / h)) return !0
                    } else if (c < g && g <= d) return !0
                }
                return !1
            }
        },
        S_a = function(a, b, c, d) {
            const e = a.minX,
                f = a.minY,
                g = a.maxX,
                h = a.maxY;
            return function(k, n, p) {
                var r = d[k * 2],
                    t = d[k * 2 + 1];
                k = d[n * 2];
                n = d[n * 2 + 1];
                var v = Math.min(t, n) - p,
                    w = Math.max(r, k) + p;
                const y = Math.max(t,
                    n) + p;
                Math.min(r, k) - p <= g && e <= w && v <= h && f <= y ? p > b ? p = !0 : (v = c.fromLatLngToPoint(new _.pn(r, t, !0)), w = c.fromLatLngToPoint(new _.pn(k, n, !0)), v = c.fromPointToLatLng(new _.Jo((v.x + w.x) / 2, (v.y + w.y) / 2), !0), t = (t + n) / 2, r = (r + k) / 2 - v.lat(), k = t - v.lng(), p = Math.sqrt(r * r + k * k) + p > b) : p = !1;
                return p
            }
        },
        U_a = function(a, b, c) {
            return function(d, e, f) {
                const g = d.path,
                    h = d.ao,
                    k = S_a(a, b, c, g);
                d = k;
                if (h) {
                    const r = R_a(h, f);
                    d = (t, v, w) => k(t, v, w) || r(t, v)
                }
                const n = [],
                    p = h ? [] : null;
                T_a(e, g.length / 2, d, function(r) {
                    n.push(g[r * 2], g[r * 2 + 1]);
                    h && p.push(h[r])
                });
                return {
                    path: n,
                    ao: p
                }
            }
        },
        W_a = function(a, b, c, d) {
            const e = a.ao;
            if (!e) return null;
            a = a.path;
            const f = [];
            for (let k = 0, n = a.length - 2; k < n;) {
                const p = a[k],
                    r = a[k + 1];
                var g = e[k / 2],
                    h = g - b;
                k += 2;
                let t = a[k],
                    v = a[k + 1];
                if (V_a(p, r, d) & V_a(t, v, d)) continue;
                for (; _.um(p, t) && _.um(r, v) && k < n;) k += 2, t = a[k], v = a[k + 1];
                const w = e[k / 2] - b,
                    y = Math.atan2(t - p, r - v);
                g === 0 && (h -= 1E-9);
                if (c)
                    for (g = Math.floor(w / c) * c; g > h;) {
                        const E = (g - h) / (w - h);
                        f.push(p + (t - p) * E, r + (v - r) * E, y);
                        g -= c
                    } else h < 0 && 0 <= w && (h = (0 - h) / (w - h), f.push(p + (t - p) * h, r + (v - r) * h, y))
            }
            return f
        },
        V_a =
        function(a, b, c) {
            let d = 0;
            _.um(a, -16, .001) && (d |= 1);
            _.um(a, c.Sh + 16, .001) && (d |= 2);
            _.um(b, -16, .001) && (d |= 4);
            _.um(b, c.Th + 16, .001) && (d |= 8);
            return d
        },
        X_a = function(a, b, c) {
            const d = [];
            if (c)
                for (let e = 0, f; f = b[e]; ++e) d.push({
                    offset: a.nh(f.offset, c),
                    repeat: a.nh(f.repeat, c)
                });
            return d
        },
        Y_a = function(a, b, c, d) {
            const e = [],
                f = [];
            for (let k = 0, n = d.length; k < n; ++k) f.push([]);
            for (let k = 0, n = b.length; k < n; ++k) {
                var g = b[k];
                let p = g.qJ;
                const r = X_a(a, d, p.ao);
                g.SE ? p = a.qh(p, g.SE, r) : p = {
                    path: p.path.slice(0),
                    ao: p.ao
                };
                if (!p.path.length) continue;
                c && (p = P_a(p, a.mh, a.ph));
                OZa(p.path, a.projection);
                p = MZa(p, a.oh);
                var h = p.path;
                g = a.fi.size;
                if (h.length) {
                    e.push(h);
                    for (let t = 0; t < h.length; t += 2) {
                        const v = _.Py(a.fi, new _.Oq(h[t], h[t + 1]), a.zoom, w => w);
                        h[t] = (v.Zh - a.Bo.x) * g.Sh;
                        h[t + 1] = (v.ai - a.Bo.y) * g.Th
                    }
                    for (let t = 0, v; v = r[t]; ++t)(h = W_a(p, v.offset, v.repeat, g)) && (f[t] = f[t].concat(h))
                }
            }
            return {
                paths: e,
                Fz: f
            }
        },
        a0a = function(a, b, c) {
            const d = Z_a(a, b, c);
            a = $_a(d, c);
            return {
                canvas: d,
                context: a,
                lH: a ? new _.zHa(a) : null,
                release: () => {
                    d.width = d.height = 0
                }
            }
        },
        Z_a = function(a, b, c) {
            a =
                a.createElement("canvas");
            a.width = b.Sh * c;
            a.height = b.Th * c;
            a.style.width = _.Em(b.Sh);
            a.style.height = _.Em(b.Th);
            _.Hq(a);
            return a
        },
        $_a = function(a, b) {
            a = a.getContext("2d");
            if (!a) return null;
            a.scale(b, b);
            a.lineCap = a.lineJoin = "round";
            return a
        },
        b0a = function(a, b, c, d, e) {
            for (let H = 0, M = a.length; H < M; ++H) {
                var f = b[H],
                    g = c,
                    h = d,
                    k = a[H],
                    n = f.CG,
                    p = f.scale,
                    r = f.rotation,
                    t = f.NM,
                    v = f.strokeColor,
                    w = f.strokeOpacity / e,
                    y = f.strokeWeight,
                    E = f.fillColor;
                f = f.fillOpacity / e;
                g.beginPath();
                for (let I = 0, F = k.length; I < F; I += 3) h.ni(n, k[I], k[I +
                    1], r + (t ? k[I + 2] : 0), p);
                f && (g.fillStyle = E, g.globalAlpha = f, g.fill());
                y && (g.lineWidth = y, g.strokeStyle = v, g.globalAlpha = w, g.stroke())
            }
        },
        CT = function(a) {
            a.Bm || (a.Bm = _.XI(() => {
                a.Bm = 0;
                const b = c0a(a);
                if (b.length) {
                    a.mh || (a.mh = a0a(a.gu, a.tileSize, a.oh), _.sx(a.mh.canvas, _.cp));
                    var c = a.mh.context;
                    a: {
                        for ({
                                fl: d
                            } of b)
                            if (d && d.vm) {
                                var d = d.vm;
                                break a
                            }
                        d = null
                    }
                    if (c) {
                        a.mh.canvas.parentNode || a.container.appendChild(a.mh.canvas);
                        c.clearRect(0, 0, a.tileSize.Sh, a.tileSize.Th);
                        for (let e = 0; e < b.length; ++e) {
                            const f = b[e],
                                g = f.rC,
                                h = g.NN;
                            if (_.um(1, h)) a.ph(c, f.paths, g.strokeColor, g.strokeOpacity, g.strokeWeight, g.strokePosition, g.Fv, g.fillColor, g.fillOpacity), b0a(f.Fz, g.nH, c, a.mh.lH, 1);
                            else {
                                a.nh || (a.nh = a0a(a.gu, a.tileSize, a.oh));
                                const k = a.nh.context;
                                k && (k.clearRect(0, 0, a.tileSize.Sh, a.tileSize.Th), a.ph(k, f.paths, g.strokeColor, g.strokeOpacity / h, g.strokeWeight, g.strokePosition, g.Fv), b0a(f.Fz, g.nH, k, a.nh.lH, h), c.globalAlpha = h, c.drawImage(a.nh.canvas, 0, 0, a.tileSize.Sh, a.tileSize.Th))
                            }
                        }
                        d && d()
                    } else d && d()
                } else a.mh && a.mh.canvas.parentNode &&
                    a.container.removeChild(a.mh.canvas), d0a(a)
            }))
        },
        c0a = function(a) {
            const b = [];
            a = a.zj;
            for (const c of Object.keys(a)) b.push(a[c]);
            b.sort((c, d) => c.zIndex - d.zIndex);
            return b
        },
        d0a = function(a) {
            a.mh && (a.mh.release(), a.mh = null);
            a.nh && (a.nh.release(), a.nh = null)
        },
        e0a = function(a, b, c) {
            function d(k, n, p) {
                p = p ? r => r : Math.round;
                k.beginPath();
                for (let r = 0, t; t = n[r]; ++r)
                    if (t.length) {
                        k.moveTo(p(t[0]), p(t[1]));
                        for (let v = 2, w = t.length; v < w;) k.lineTo(p(t[v++]), p(t[v++]))
                    }
            }
            const e = c.size,
                f = a.createElement("canvas");
            f.width = b * e.Sh;
            f.height = b * e.Th;
            const g = f.getContext("2d");
            g.lineCap = g.lineJoin = "round";
            g.scale(b, b);
            const h = [function(k, n, p, r, t) {
                k.lineWidth = t;
                k.strokeStyle = p;
                k.globalAlpha = r;
                k.stroke()
            }, function(k, n, p, r, t) {
                k.lineWidth = t * 2;
                k.strokeStyle = p;
                k.globalAlpha = r;
                k.save();
                k.clip();
                k.stroke();
                k.restore()
            }, function(k, n, p, r, t, v) {
                g.lineWidth = t * 2;
                g.strokeStyle = p;
                g.globalCompositeOperation = "source-over";
                g.clearRect(0, 0, e.Sh, e.Th);
                d(g, n, v);
                g.stroke();
                g.globalCompositeOperation = "destination-out";
                g.fill();
                k.globalAlpha = r;
                k.drawImage(f,
                    0, 0, e.Sh, e.Th)
            }];
            return function(k, n, p, r, t, v, w, y, E) {
                d(k, n, w);
                y != null && E && (k.fillStyle = y, k.globalAlpha = E, k.fill());
                if (t) h[v](k, n, p, r, t, w)
            }
        },
        g0a = function(a, b, c) {
            a.mh = a.mh || e0a(a.gu, b, a.fi);
            return new f0a(a.gu, c, a.fi, a.mh, b)
        },
        i0a = function(a, b) {
            b.kC = () => {
                Object.values(b.geometry.zj).forEach(d => {
                    d.rC = b.style;
                    d.zIndex = b.style.zIndex;
                    CT(d.fl.sC)
                })
            };
            const c = b.geometry.bounds;
            c.poly = b;
            b.bounds = c;
            _.wM(a.nh, c);
            a = a.mh.search(c);
            for (let d = 0, e = a.length; d < e; ++d) h0a(b, a[d].sj)
        },
        j0a = function(a, b) {
            delete b.kC;
            a.nh.remove(b.bounds);
            a = b.geometry;
            Object.values(a.zj).forEach(c => {
                const d = c.fl;
                var e = d.sC;
                delete e.zj[_.Zn(c)];
                CT(e);
                delete d.zj[_.Zn(c)]
            });
            a.zj = {}
        },
        k0a = function(a, b) {
            const c = b.bounds;
            c.sj = b;
            _.wM(a.mh, c);
            a = a.nh.search(b.bounds);
            if (a.length === 0) b.vm && _.mq(b.vm.bind(b));
            else
                for (let d = 0, e = a.length; d < e; ++d) h0a(a[d].poly, b)
        },
        l0a = function(a, b) {
            a.mh.remove(b.bounds);
            Object.values(b.zj).forEach(c => {
                delete c.GG.zj[_.Zn(c)]
            });
            b.zj = {}
        },
        h0a = function(a, b) {
            const c = a.geometry;
            var d = Y_a(b.RN, c.NL, c.geodesic, c.symbols);
            d.paths.length ?
                (a = {
                    paths: d.paths,
                    Fz: d.Fz,
                    rC: a.style,
                    zIndex: a.style.zIndex,
                    fl: b,
                    GG: c
                }, d = _.Zn(a), c.zj[d] = a, b.zj[d] = a, b = b.sC, b.zj[_.Zn(a)] = a, CT(b)) : b.vm && _.mq(b.vm.bind(b))
        },
        n0a = function(a) {
            return (a || []).map(b => {
                const c = b.path;
                let d = Infinity,
                    e = Infinity,
                    f = -Infinity,
                    g = -Infinity;
                for (let k = 0, n = c.length; k < n;) {
                    var h = c[k++];
                    d = Math.min(d, h);
                    f = Math.max(f, h);
                    h = c[k++];
                    e = Math.min(e, h);
                    g = Math.max(g, h)
                }
                h = new _.rp;
                h.minX = d;
                h.maxX = f;
                h.minY = e;
                h.maxY = g;
                return {
                    qJ: b,
                    bounds: h,
                    ...(c.length >= 50 && {
                        SE: m0a(c)
                    })
                }
            })
        },
        q0a = function(a, b, c, d) {
            return a.map(e => {
                const f = e.path;
                e = e.ao;
                var g = null;
                if (b) {
                    g = c ? o0a(f) : p0a(f, d);
                    const h = new Map;
                    h.set(0, 0);
                    let k = 0;
                    const n = [0];
                    for (let p = 1, r = e.length; p < r; ++p) {
                        const t = e[p];
                        h.has(t) ? k = h.get(t) : (k += g(p - 1), h.set(t, k));
                        n.push(k)
                    }
                    g = n
                }
                return {
                    path: f,
                    ao: g
                }
            })
        },
        p0a = function(a, b) {
            const c = a.slice(0);
            OZa(c, b);
            return d => {
                d *= 2;
                const e = c[d] - c[d + 2];
                d = c[d + 1] - c[d + 3];
                return Math.sqrt(e * e + d * d)
            }
        },
        o0a = function(a) {
            return b => {
                b *= 2;
                const c = new _.pn(a[b], a[b + 1]);
                b = new _.pn(a[b + 2], a[b + 3]);
                return _.vI(c, b, 1)
            }
        },
        u0a = function(a) {
            a.length && (a = r0a(a,
                s0a), a = r0a(a, t0a));
            return a
        },
        r0a = function(a, b) {
            const c = [];
            let d = new tT(0);
            const e = new tT(0);
            let f = new tT(0);
            const g = Array(a.length);
            g[0] = a[0];
            g[1] = a[1];
            eT(a, d);
            for (let h = 2, k = 2; h < a.length;) {
                c[0] = a[h];
                c[1] = a[h + 1];
                eT(c, f);
                b(d, f, e) && (fT(e, c), g[k++] = c[0], g[k++] = c[1]);
                g[k++] = a[h++];
                g[k++] = a[h++];
                const n = d;
                d = f;
                f = n
            }
            return g
        },
        s0a = function(a, b, c) {
            if (a.z > 0 === b.z > 0) return !1;
            dT(a, b, DT);
            dT(DT, v0a, c);
            c.z = 0;
            if (cT(c, c) < 1E-12) return !1;
            cT(c, a) + cT(c, b) < 0 && (c.x = -c.x, c.y = -c.y);
            return !0
        },
        t0a = function(a, b, c) {
            dT(a, b, DT);
            dT(v0a, DT, ET);
            dT(DT, ET, c);
            if (cT(c, c) < 1E-12 || cT(a, ET) > 0 === cT(b, ET) > 0) return !1;
            cT(c, a) + cT(c, b) < 0 && (c.x = -c.x, c.y = -c.y, c.z = -c.z);
            return !0
        },
        w0a = function(a, b, c) {
            if (!b) return null;
            const d = [];
            b.forEach(e => {
                var f = d.push;
                e = e.mh;
                const g = e.length,
                    h = Array(g * 2);
                for (let k = 0, n = 0; k < g; ++k) {
                    const p = e[k];
                    h[n++] = p.lat();
                    h[n++] = p.lng()
                }
                f.call(d, h)
            });
            if (a)
                for (const e of d) NZa(e);
            if (c)
                for (let e = 0, f = d.length; e < f; ++e) d[e] = u0a(d[e]);
            return d
        },
        x0a = function(a) {
            return (a || []).map(b => {
                if (b.length) {
                    var c = b[1];
                    var d = [0];
                    for (let e =
                            1, f = 1; e < b.length / 2; ++e, ++f) {
                        const g = b[2 * e + 1];
                        if (Math.abs(c - g) > 180) {
                            const h = g < c ? 1 : -1,
                                k = b[2 * e - 2],
                                n = b[2 * e];
                            b.splice(2 * e, 0, n, g + 360 * h, n, g + 450 * h, 90, g + 450 * h, 90, c - 450 * h, k, c - 450 * h, k, c - 360 * h);
                            d.push(f, f, f, f, f, f - 1);
                            e += 6
                        }
                        d.push(f);
                        c = g
                    }
                    c = d
                } else c = [];
                return {
                    path: b,
                    ao: c
                }
            })
        },
        y0a = function(a) {
            return a.map(b => P_a(b, _.rt, .1))
        },
        x_a = function(a, b) {
            var c = a.get("icons"),
                d = !a.Gn && !!_.pm(c);
            let e = !0;
            c = d ? c.map(k => {
                const n = RZa(k.offset) || z0a;
                k = RZa(k.repeat) || A0a;
                e = e && (n.value === 0 || n.unit === "%") && (k.value === 0 || k.unit === "%");
                return {
                    offset: n,
                    repeat: k
                }
            }) : [];
            const f = a.get("geodesic");
            var g = w0a(a.Gn, a.get("latLngs"), f);
            g = x0a(g);
            const h = f && e;
            f && d && !h && (g = y0a(g));
            g = q0a(g, d, h, b);
            b = n0a(g);
            if (b.length === 1) d = b[0].bounds;
            else {
                d = new _.rp;
                for (let k = 0, n = b.length; k < n; ++k) d.extendByBounds(b[k].bounds)
            }
            return {
                bounds: d,
                KA: a,
                geodesic: f,
                NL: b,
                zj: {},
                symbols: c
            }
        },
        w_a = function(a, b) {
            const c = b.Gn,
                d = t => {
                    const v = b.get(t);
                    return c ? _.ym(v, a.mh[t]) : _.ym(v, a.oh[t])
                },
                e = (b.get("zIndex") || 0) * 1E3 + _.Ka(b) % 1E3,
                f = c ? d("strokePosition") : a.mh.strokePosition;
            var g = f === 0 ? 32 : 16;
            const h = Math.min(d("strokeWeight"), g),
                k = d("strokeColor"),
                n = d("strokeOpacity");
            g = [];
            let p = 1;
            if (!c) {
                const t = b.get("icons") || [];
                g = t.map(w => {
                    const y = a.nh(w.icon || {}, k, n, h);
                    y.NM = !w.fixedRotation;
                    return y
                });
                let v = 0;
                for (var r of g) v = Math.max(v, r.strokeOpacity), v = Math.max(v, r.fillOpacity);
                _.um(0, v) || (_.um(0, n) || _.um(0, h)) && g.length === 1 || (p = Math.max(v, n));
                _.pm(t) && (r = b.get("map"), _.N(r, 147754))
            }
            if (!(r = b.get("hitStrokeWeight"))) a: switch (f) {
                case 1:
                    r = 0;
                    break a;
                case 2:
                    r = h * 2;
                    break a;
                default:
                    r = h
            }
            return {
                strokeColor: k,
                strokeOpacity: n,
                strokeWeight: h,
                strokePosition: f,
                dL: r,
                ...(c && {
                    fillColor: d("fillColor"),
                    fillOpacity: d("fillOpacity")
                }),
                zIndex: e,
                clickable: d("clickable"),
                nH: g,
                NN: p,
                Gn: c,
                Fv: !!b.Fv
            }
        },
        G0a = function(a, b) {
            a.nh = new B0a(a.tiles, a.ui, b.wk, a.ph);
            a.ui.sh.register(a.nh);
            a.mh = C0a(a, b);
            _.Rl("geometry").then(c => {
                a.geometry = c;
                a.oh = new rT((d, e, f) => d.get("geodesic") && a.geometry != null ? new D0a(d, e, f, a.geometry.spherical.computeHeading, a.geometry.spherical.computeOffset, a.geometry.spherical.computeOffsetOrigin) : new E0a(d,
                    e, f), a.map, b);
                mT(a.map, a.oh);
                a.ui.uh.forEach(d => {
                    F0a(a, d)
                })
            })
        },
        F0a = function(a, b) {
            a.oh && a.mh && a.nh && a.geometry && a.Mh && (nT(b, a.oh), I_a(b, a.map, a.mh, a.geometry.spherical.interpolate, a.Mh), q_a(b, a.ui, a.nh, a.ui.sh), b.rh || _.N(a.map, b.Gn ? 147752 : 147753))
        },
        C0a = function(a, b) {
            b = new FT(b);
            b.bindTo("panes", a.ui);
            b.bindTo("projection", a.map);
            b.bindTo("zoom", a.ui);
            b.bindTo("projectionCenterQ", a.ui);
            b.bindTo("offset", a.ui);
            _.Rn(a.map, "click", b, b.Pk);
            return b
        },
        I0a = function(a) {
            const b = a.getVisible() != 0 ? a.getMap() :
                null;
            a.qh !== b && (a.qh && a.qh.__gm.uh.remove(a), b && (b.__gm.uh.mh || new H0a(b), _.rq(b.__gm.uh, a)), a.qh = b)
        },
        J0a = function(a, b, c) {
            if (!a || !b || !c) return null;
            var d = c.lat();
            const e = c.lng();
            c = b & 1 ? d : a.bj.lo;
            let f = b & 2 ? e : a.xi.lo;
            d = b & 4 ? d : a.bj.hi;
            let g = b & 8 ? e : a.xi.hi;
            if (c > d) {
                var h = c;
                c = d;
                d = h
            }
            b & 10 && !a.xi.contains(e) && (h = _.ro(e, a.xi.lo), a = _.ro(a.xi.hi, e), b & 2 ? h > a : a > h) && (b = f, f = g, g = b);
            return _.zo(c, f, d, g)
        },
        GT = function(a) {
            a && (a.unbindAll(), a.release(), _.Mn(a))
        },
        Q0a = function(a, b, c) {
            let d;
            d = c ? new iT(a.nh, K0a, L0a, !1, b.overlayMouseTarget,
                a.th, a.get("map"), a.sh) : new iT(a.qh, M0a, N0a, !0, b.overlayMouseTarget, a.th, a.get("map"), a.sh);
            d.bindTo("projection", a);
            d.bindTo("zoom", a);
            d.bindTo("projectionCenterQ", a);
            d.bindTo("panningEnabled", a);
            d.bindTo("mapPixelBounds", a);
            d.bindTo("color", a);
            d.bindTo("zIndex", a);
            d.bindTo("offset", a);
            const e = c ? O0a : P0a;
            let f;
            _.Hn(d, "dragstart", g => {
                f = a.get("bounds");
                a.mh.bindTo("freeVertexPosition", d);
                a.mh.set("freeControlPoint", e[g]);
                a.mh.set("map", a.get("map"))
            });
            _.Hn(d, "dragend", (g, h, k) => {
                a.mh.set("map", null);
                _.Wn(a, "toolbar", {
                    show: !0,
                    Ex: k,
                    action: () => {
                        a.set("bounds", f)
                    }
                })
            });
            _.Vn(d, "dragstart", a);
            _.Vn(d, "dragend", a);
            _.Vn(d, "panbynow", a);
            return d
        },
        R0a = function(a, b, c) {
            a.oh || (a.oh = !0, b = J0a(a.get("bounds"), b, c), a.set("bounds", b), a.oh = !1)
        },
        U0a = function(a, b, c) {
            function d() {
                a.get("editable") ? S0a(a, b, c) : (T0a(a), _.Wn(a, "toolbar", {
                    show: !1
                }))
            }
            a.editable_changed = d;
            d()
        },
        S0a = function(a, b, c) {
            if (!a.oh) {
                var d = new _.oN(a, !0);
                a.qh = d;
                var e = new V0a;
                e.bindTo("strokeColor", d);
                e.bindTo("strokeOpacity", d, "ghostStrokeOpacity");
                e.bindTo("strokeWeight", d);
                e.bindTo("bounds", a);
                e.bindTo("zIndex", a);
                a.ph = e;
                var f = b.__gm;
                a.mh = bT(f);
                var g = _.vM(f, a),
                    h = _.wx() ? 9 : 0,
                    k = new kT;
                c = new W0a(e, function(n, p, r) {
                    return new lT(n, a.mh, h, p, g, k, r)
                }, c);
                c.set("map", b);
                c.bindTo("bounds", a);
                c.bindTo("panes", f);
                c.bindTo("projection", b);
                c.bindTo("zoom", f);
                c.bindTo("projectionCenterQ", f);
                c.bindTo("panningEnabled", b, "draggable");
                c.bindTo("mapPixelBounds", f, "pixelBounds");
                c.bindTo("offset", f);
                c.bindTo("color", d, "strokeColor");
                c.bindTo("zIndex", a);
                c.bindTo("suppressGhostControlPoints",
                    a);
                a.oh = c;
                _.Vn(c, "panbynow", f);
                _.Vn(c, "toolbar", a)
            }
        },
        T0a = function(a) {
            const b = a.oh;
            b && (b.unbindAll(), b.set("map", null), _.Mn(b), delete a.oh, a.mh ? .unbindAll(), a.mh ? .release(), delete a.mh, a.ph ? .unbindAll(), delete a.ph, a.qh ? .release(), delete a.qh)
        },
        Z0a = function(a, b) {
            a.nh = new rT((c, d, e) => new X0a(c, d, e), a.mh, b);
            mT(a.mh, a.nh);
            a.ui.yh.forEach(c => {
                Y0a(a, c)
            })
        },
        Y0a = function(a, b) {
            if (a.nh && a.Mh) {
                var c = b.rh = new $0a(b);
                c.set("map", a.mh);
                c.bindTo("bounds", b);
                c.bindTo("capturing", b);
                c.bindTo("cursor", b);
                c.bindTo("clickable",
                    b);
                c.bindTo("fillColor", b);
                c.bindTo("fillOpacity", b);
                c.bindTo("strokeColor", b);
                c.bindTo("strokeOpacity", b);
                c.bindTo("strokeWeight", b);
                c.bindTo("strokePosition", b);
                c.bindTo("suppressUndo", b);
                c.bindTo("zIndex", b);
                var d = [];
                b.Wh = d;
                sT.forEach(e => {
                    d.push(_.Vn(c, e, b))
                });
                d.push(_.Vn(b, "toolbar", c));
                nT(b, a.nh);
                U0a(b, a.mh, a.Mh);
                _.N(a.mh, 147755)
            }
        },
        b1a = function(a) {
            const b = a.getVisible() != 0 ? a.getMap() : null;
            a.nh !== b && (a.nh && a.nh.__gm.yh.remove(a), b && (b.__gm.yh.mh || new a1a(b), _.rq(b.__gm.yh, a)), a.nh = b)
        },
        sT = "click dblclick mousedown mousemove mouseout mouseover mouseup rightclick contextmenu".split(" "),
        l_a = class extends _.$n {
            constructor(a) {
                super();
                this.nh = [];
                const b = new _.fu;
                this.poly = b;
                b.mh = a;
                b.rh = !0;
                b.bindTo("capturing", this);
                b.bindTo("cursor", this);
                b.bindTo("map", this);
                b.bindTo("strokeColor", this);
                b.bindTo("strokeOpacity", this);
                b.bindTo("strokeWeight", this);
                b.bindTo("strokePosition", this);
                b.bindTo("fillColor", this);
                b.bindTo("fillOpacity", this);
                b.bindTo("clickable", this);
                b.bindTo("zIndex", this);
                b.bindTo("suppressUndo", this);
                b.Fv = !0;
                const c = this.nh;
                sT.forEach(d => {
                    c.push(_.Vn(b, d, this))
                });
                c.push(_.Vn(this,
                    "toolbar", b));
                this.mh = new _.oq(() => {
                    JZa(this)
                }, 0)
            }
            center_changed() {
                this.mh.ri()
            }
            radius_changed() {
                this.mh.ri()
            }
            planetRadius_changed() {
                this.mh.ri()
            }
            release() {
                const a = this.nh;
                for (let b = 0, c = a.length; b < c; ++b) _.Jn(a[b]);
                this.nh = [];
                this.poly.unbindAll();
                this.poly = null
            }
        },
        QZa = {};
    var tT = class {
        constructor(a) {
            this.y = this.x = 0;
            this.z = a
        }
        equals(a) {
            return this.x === a.x && this.y === a.y && this.z === a.z
        }
    };
    var iT = class extends _.$n {
        constructor(a, b, c, d, e, f, g, h) {
            super();
            this.wh = a;
            this.rh = b;
            this.qh = c;
            this.oh = d;
            this.uh = f;
            this.map = g;
            this.Mh = h;
            this.mh = [];
            this.container = _.tx("div", e, _.cp);
            a.forEach(this.ph.bind(this));
            const k = () => {
                this.oh || _.Wn(this, "toolbar", {
                    show: !1
                })
            };
            this.sh = [_.Hn(a, "insert_at", n => {
                this.ph(a.getAt(n), n);
                VZa(this, n + 1);
                k()
            }), _.Hn(a, "remove_at", n => {
                const p = this.mh[n];
                this.mh.splice(n, 1);
                this.th(p);
                VZa(this, n);
                k()
            }), _.Hn(a, "set_at", n => {
                this.mh[n].Cp.set("latLngPosition", a.getAt(n));
                k()
            })]
        }
        release() {
            this.sh.forEach(_.Jn);
            this.sh.length = 0;
            this.mh.forEach(this.th.bind(this));
            this.mh.length = 0;
            _.Mn(this.container);
            _.Dl(this.container)
        }
        zIndex_changed() {
            let a = this.get("zIndex") || 0;
            this.oh && --a;
            _.ux(this.container, a)
        }
        ph(a, b) {
            const c = {};
            c.index = b;
            this.mh.splice(b, 0, c);
            c.constraint = this.qh[b % this.qh.length];
            const d = this.uh(this.container, this.rh[b % this.rh.length], this.oh);
            d.bindTo("color", this);
            d.bindTo("panningEnabled", this);
            d.bindTo("mapPixelBounds", this);
            c.Cp = d;
            b = [_.Vn(d, "panbynow", this)];
            const e = a && this.map ? _.Cx(a, this.map.getProjection()) :
                null;
            d.set("latLngPosition", a);
            c.ow = !0;
            c.rq = null;
            c.eu = null;
            a = new _.lN(this.Mh.wk, {
                kp: f => {
                    f ? d.set("position", new _.Jo(f.Sh, f.Th)) : d.set("position", null)
                },
                zt: () => {}
            }, e, this.Mh, () => {
                if (this.map) {
                    var f = c.Kk.getPosition();
                    f && (f = _.Ar(f, this.map.getProjection()), c.ow = !1, c.Cp.set("latLngPosition", f), c.ow = !0)
                }
            });
            this.Mh.Cj(a);
            c.Kk = a;
            b.push(_.Hn(d, "latlngposition_changed", () => {
                if (c.ow && this.map) {
                    var f = c.Cp.get("latLngPosition");
                    f ? c.Kk.setPosition(_.Cx(f, this.map.get("projection"))) : c.Kk.setPosition(null)
                }
            }));
            b.push(_.Hn(d, "dragstart", () => {
                this.qn(c)
            }));
            b.push(_.Hn(d, "dragend", () => {
                this.Ln(c)
            }));
            b.push(_.Hn(d, "deltaclientposition_changed", () => {
                this.pn(c)
            }));
            b.push(_.Hn(d, "dragstart", f => {
                this.bindTo("freeVertexPosition", d, "latLngPosition");
                gT(this, c, "dragstart", f)
            }));
            b.push(_.Hn(d, "dragend", f => {
                this.wh.setAt(c.index, d.get("latLngPosition"));
                gT(this, c, "dragend", f)
            }));
            for (const f of sT) b.push(_.Hn(d, f, g => {
                gT(this, c, f, g)
            }));
            c.Wh = b
        }
        qn(a) {
            a.rq = a.Kk.ho();
            a.rq && _.EM(a.Kk, a.rq)
        }
        Ln(a) {
            var b = a.Kk.getPosition();
            if (b &&
                this.map && (b = _.Ar(b, this.map.get("projection")), b = WZa(this, a, b)) && (b = _.Cx(b, this.map.get("projection")), b = a.Kk.ho(b), b = XZa(a, b)) && (b = a.Kk.getPosition(b))) {
                const c = _.Ar(b, this.map.get("projection"));
                a.ow = !1;
                a.Cp.set("latLngPosition", c);
                a.ow = !0;
                a.Kk.setPosition(b)
            }
            a.rq = null;
            a.eu = null
        }
        pn(a) {
            var b = a.Cp.get("deltaClientPosition");
            if (b && (a.rq || a.eu) && this.map) {
                var c = a.eu || a.rq;
                a.eu = {
                    clientX: c.clientX + b.clientX,
                    clientY: c.clientY + b.clientY
                };
                b = this.Mh.Nm(a.eu);
                b = _.Ar(b, this.map.get("projection"));
                c = a.eu;
                var d =
                    WZa(this, a, b);
                d && !d.equals(b) && (b = _.Cx(d, this.map.get("projection")), c = a.Kk.ho(b));
                (c = XZa(a, c)) && _.EM(a.Kk, c)
            }
        }
        th(a) {
            a.Cp.unbindAll();
            a.Cp.release();
            a.Kk && this.Mh.Ql(a.Kk);
            a.Wh.forEach(_.Jn);
            a.Wh.length = 0
        }
    };
    var g_a = class extends _.$n {
            center_changed() {
                hT(this)
            }
            radius_changed() {
                hT(this)
            }
            planetRadius_changed() {
                hT(this)
            }
            constructor(a, b, c) {
                super();
                this.mh = a;
                this.Mh = c;
                this.qh = null;
                this.ph = this.oh = !1;
                this.rh = b;
                this.nh = new _.xp;
                _.Rn(this.nh, "set_at", this, this.sh)
            }
            map_changed() {
                this.get("map") || (this.mh.set("map", null), YZa(this), _.Mn(this.nh))
            }
            panes_changed() {
                YZa(this);
                this.qh = a_a(this)
            }
            sh(a) {
                if (!this.ph) {
                    var b = this.nh.getAt(a);
                    if (a === 0) _.WI(this, function() {
                        this.oh = !0;
                        this.set("center", b);
                        this.oh = !1
                    }, 0);
                    else {
                        const c =
                            _.vI(this.get("center"), b, this.get("planetRadius"));
                        _.WI(this, function() {
                            this.oh = !0;
                            this.set("radius", c);
                            this.oh = !1
                        }, 0)
                    }
                }
            }
        },
        ZZa = ["pointer", "row-resize", "row-resize", "col-resize", "col-resize"],
        $Za = [0, 2, 2, 1, 1];
    var kT = class {
        construct(a, b, c, d, e) {
            a = _.tx("div", a, new _.Jo(b, b), new _.No(c, c));
            a.style.borderWidth = _.Em(d);
            a.style.borderStyle = "solid";
            a.style.borderRadius = _.Em(Math.ceil(c / 2 + d));
            _.eJ(a, e)
        }
        setFillColor(a, b) {
            a.firstChild.style.backgroundColor = b
        }
    };
    var f_a = class extends _.$n {
        constructor() {
            super();
            const a = new _.Ap({
                clickable: !1,
                fillOpacity: 0
            });
            a.bindTo("map", this);
            a.bindTo("strokeColor", this);
            a.bindTo("strokeOpacity", this);
            a.bindTo("strokeWeight", this);
            var b = ["center", "freeVertexPosition", "freeVertexIsCenter"],
                c = new _.PM(b, "return", b_a);
            _.$I(c, b, this);
            a.bindTo("center", c, "return");
            b = ["center", "radius", "planetRadius", "freeVertexPosition", "freeVertexIsCenter"];
            c = new _.PM(b, "return", (d, e, f, g, h) => !h && g && d ? _.vI(g, d, f) : e);
            _.$I(c, b, this);
            a.bindTo("radius",
                c, "return");
            this.mh = _.QM();
            this.mh.bindTo("zIndex", this);
            a.bindTo("zIndex", this.mh, "ghostZIndex")
        }
    };
    var lT = class extends _.$n {
        constructor(a, b, c, d, e, f, g) {
            super();
            this.Hh = e;
            this.nh = !1;
            a = _.tx("div", a);
            a.style.display = "none";
            e = 9 + 2 * (1 + c);
            _.Eq(a, new _.No(e, e));
            f.construct(a, c, 9, 1, g ? .5 : 1);
            g = new _.AHa(a, !0);
            g.bindTo("containerPixelBounds", this, "mapPixelBounds");
            g.bindTo("panningEnabled", this);
            g.set("draggableCursor", d);
            g.bindTo("position", this);
            g.bindTo("deltaClientPosition", this);
            this.mh = a;
            this.rh = b;
            this.th = c;
            this.ph = g;
            this.uh = f;
            this.oh = [_.Hn(g, "dragstart", () => {
                b.set("draggableCursor", d);
                b.set("active", !0)
            }), _.Hn(g, "dragend", () => {
                b.set("active", !1)
            }), _.Vn(g, "panbynow", this), _.Vn(g, "dragstart", this), _.Vn(g, "dragend", this)];
            this.qh = _.jy(a, {
                zl: h => {
                    _.Vx(h);
                    _.Wn(this, "mousedown", h.mh)
                },
                Or: h => {
                    _.Wx(h);
                    _.Wn(this, "mousemove", h.mh)
                },
                wm: h => {
                    _.Wx(h);
                    _.Wn(this, "mousemove", h.mh)
                },
                Ol: h => {
                    _.Xx(h);
                    _.Wn(this, "mouseup", h.mh)
                },
                nl: ({
                    event: h,
                    Fq: k
                }) => {
                    _.Zw(h.mh);
                    h.button === 3 ? k || _.Wn(this, "rightclick", h.mh) : k ? _.Wn(this, "dblclick", h.mh) : _.Wn(this, "click", h.mh)
                },
                zu: h => {
                    _.Yx(h);
                    _.Wn(this, "contextmenu", h.mh)
                }
            });
            this.sh = new _.LB(a,
                a, {
                    tt: h => {
                        this.nh = !1;
                        jT(this);
                        _.En(h);
                        _.Fn(h);
                        _.Wn(this, "mouseout", h)
                    },
                    ut: h => {
                        this.nh = !0;
                        jT(this);
                        _.En(h);
                        _.Fn(h);
                        _.Wn(this, "mouseover", h)
                    }
                })
        }
        release() {
            _.Dl(this.mh);
            this.oh.forEach(_.Jn);
            this.oh.length = 0;
            this.qh.remove();
            this.sh.remove();
            this.ph.unbindAll();
            this.ph.release();
            this.rh.set("active", !1)
        }
        position_changed() {
            const a = this.get("position");
            if (a) {
                this.mh.style.display = "block";
                var b = 5 + this.th;
                _.sx(this.mh, new _.Jo(a.x - b, a.y - b))
            } else this.mh.style.display = "none"
        }
        color_changed() {
            const a = this.get("color");
            this.mh.firstChild.style.borderColor = a;
            jT(this)
        }
    };
    var i_a = class {
            constructor(a, b, c, d, e) {
                this.oh = a;
                this.nh = c;
                a = a.get("center");
                b = c.fromDivPixelToLatLng(b);
                this.ph = d(a, b);
                this.mh = _.vI(b, a);
                this.computeOffsetOrigin = e
            }
            moveTo(a) {
                a = this.nh.fromDivPixelToLatLng(a);
                var b = this.computeOffsetOrigin(a, this.mh, this.ph);
                b || (b = _.vI(a, c1a) - this.mh, a = _.vI(a, d1a) - this.mh, b = Math.abs(b) < Math.abs(a) ? c1a : d1a);
                this.oh.set("center", b)
            }
        },
        c1a = new _.pn(90, 0),
        d1a = new _.pn(-90, 0);
    var HT = new _.No(30, 30),
        e1a = new _.Jo(HT.width / 2, HT.height / 2),
        rT = class extends _.$n {
            constructor(a, b, c) {
                super();
                this.Ah = a;
                this.nh = this.ph = this.wh = this.th = this.mh = null;
                this.uh = !1;
                this.rh = b;
                this.oh = new _.lN(c.wk, {
                    kp: d => {
                        d && this.set("position", new _.Jo(d.Sh, d.Th))
                    },
                    zt: () => {}
                }, null, c);
                c.Cj(this.oh);
                this.qh = new _.jN((d, e) => {
                    _.Wn(this, "panbynow", d, e)
                });
                a = new _.PM(["waitingForQuiver", "dragging"], "active", (d, e) => d || e);
                a.bindTo("dragging", this, null, !0);
                a.bindTo("waitingForQuiver", this, null, !0);
                this.bindTo("active",
                    a, null, !0)
            }
            release() {
                this.qh.release();
                this.nh && this.nh.forEach(_.Jn);
                this.nh = null
            }
            yh(a) {
                if (this.get("draggable") && this.mh && (!a.domEvent || !_.iv(a.domEvent)) && (a.stop(), !this.get("dragging") && a.latLng)) {
                    this.sh = a;
                    this.set("position", this.ph.fromLatLngToDivPixel(a.latLng));
                    var b = _.Cx(a.latLng, this.rh.getProjection());
                    (b = this.oh.ho(b)) && _.EM(this.oh, b);
                    b = this.get("position");
                    this.set("dragging", !0);
                    this.set("waitingForQuiver", !1);
                    this.wh = this.Ah(this.mh, b, this.ph);
                    qT(this, "dragstart", a);
                    this.uh = this.mh.get("editable");
                    this.mh.set("editable", !1);
                    this.set("storeEditable", !0)
                }
            }
            zh(a) {
                this.get("dragging") && (this.sh = a, a.latLng && (a = _.Cx(a.latLng, this.rh.getProjection()), (a = this.oh.ho(a)) && _.EM(this.oh, a)))
            }
            xh(a) {
                if (this.get("dragging")) {
                    var b = this.oh.getPosition();
                    if (b) {
                        var c = _.Ar(b, this.rh.getProjection());
                        c = this.ph.fromLatLngToDivPixel(c);
                        this.sh = a;
                        this.set("position", c);
                        this.oh.setPosition(b)
                    }
                    this.set("storeEditable", !1);
                    this.mh.set("editable", this.uh);
                    qT(this, "dragend", a);
                    this.set("dragging", !1)
                }
            }
            position_changed() {
                const a =
                    this.get("position");
                a ? _.AM(this.qh, _.iGa(a, HT, e1a)) : _.AM(this.qh, null);
                this.get("dragging") && (this.wh.moveTo(a), qT(this, "drag", this.sh))
            }
            containerPixelBounds_changed() {
                _.CM(this.qh, this.get("containerPixelBounds"))
            }
            Bh() {
                this.get("dragging") || this.set("waitingForQuiver", !0)
            }
            Ch() {
                this.set("waitingForQuiver", !1)
            }
            active_changed() {
                pT(this)
            }
            panes_changed() {
                pT(this)
            }
            poly_changed() {
                pT(this)
            }
            projectionController_changed() {
                pT(this)
            }
            dragging_changed() {
                h_a(this)
            }
            panningEnabled_changed() {
                h_a(this)
            }
        };
    var m_a = class {
        constructor(a) {
            this.map = a;
            this.Mh = this.mh = null;
            this.ui = a.__gm;
            this.ui.nh.then(({
                Mh: b
            }) => {
                this.Mh = b;
                k_a(this, b)
            });
            this.ui.xh.mh = b => {
                j_a(this, b)
            };
            this.ui.xh.onRemove = b => {
                this.onRemove(b)
            }
        }
        onRemove(a) {
            if (this.Mh) {
                var b = a.rh;
                b.unbindAll();
                b.set("map", null);
                b.release();
                delete a.rh;
                a.Wh && a.Wh.forEach(_.Jn);
                delete a.Wh;
                oT(a);
                delete a.editable_changed;
                d_a(a)
            }
        }
    };
    var D0a = class {
        constructor(a, b, c, d, e, f) {
            this.poly = a;
            this.ph = c;
            this.computeOffset = e;
            this.computeOffsetOrigin = f;
            const g = o_a(this);
            b = c.fromDivPixelToLatLng(b);
            this.qh = _.vI(g, b);
            this.rh = d(g, b);
            const h = [];
            this.oh = h;
            a.get("latLngs").forEach(k => {
                const n = [];
                k.forEach(p => {
                    n.push({
                        heading: d(g, p),
                        distance: _.vI(p, g)
                    })
                });
                h.push(n)
            });
            this.mh = g;
            this.nh = b.lng() - g.lng()
        }
        moveTo(a) {
            a = this.ph.fromDivPixelToLatLng(a);
            let b = this.computeOffsetOrigin(a, this.qh, this.rh);
            b ? Math.abs(b.lat()) > 89 && (b = new _.pn(_.sm(b.lat(), -89,
                89), b.lng())) : b = new _.pn(this.mh.lat(), a.lng() - this.nh);
            this.poly.get("latLngs").forEach((c, d) => {
                for (let e = 0, f = c.getLength(); e < f; ++e) {
                    const g = this.computeOffset(b, this.oh[d][e].distance, this.oh[d][e].heading);
                    c.setAt(e, g)
                }
            });
            this.mh = b;
            this.nh = a.lng() - b.lng()
        }
    };
    var p_a = class extends _.$n {
        constructor() {
            super(...arguments);
            this.zIndex = Infinity
        }
        Ft(a) {
            return a !== "dragstart" && a !== "drag" && a !== "dragend"
        }
        Nt() {
            return this.get("active") ? this : null
        }
        handleEvent(a, b, c) {
            if (a === "mouseout") this.set("cursor", "");
            else if (a === "mouseover") {
                const d = this.get("draggableCursor");
                d && this.set("cursor", d)
            }
            _.Wn(c, a, new _.MB(b.latLng, b.domEvent))
        }
    };
    var s_a = {
            latLngs: 1,
            geodesic: 1,
            icons: 1
        },
        t_a = {
            strokeColor: 1,
            strokeOpacity: 1,
            strokePosition: 1,
            strokeWeight: 1,
            fillColor: 1,
            fillOpacity: 1,
            hitStrokeWeight: 1,
            zIndex: 1,
            clickable: 1,
            icons: 1
        },
        g1a = class extends _.$n {
            constructor(a, b) {
                var c = new f1a;
                super();
                this.qh = a;
                this.oh = b;
                this.ph = c;
                this.nh = {};
                this.mh = {};
                this.timeout = 0;
                const d = this;
                this.rh = function(e) {
                    v_a(d, this, e)
                };
                a.mh = e => {
                    this.Cu(e)
                };
                a.onRemove = e => {
                    this.Cw(e)
                }
            }
            projection_changed() {
                const a = this;
                a.qh.forEach(b => {
                    if (b.get("icons")) {
                        const c = _.Zn(b);
                        a.mh[c] = b;
                        vT(b)
                    }
                });
                u_a(a)
            }
            Cu(a) {
                var b = this.get("projection");
                b = a.ph = {
                    style: w_a(this.ph, a),
                    geometry: x_a(a, b)
                };
                y_a(this, a);
                a.changed = this.rh;
                _.rq(this.oh, b)
            }
            Cw(a) {
                const b = a.ph;
                delete a.ph;
                this.oh.remove(b);
                delete a.changed;
                vT(a);
                a = _.Zn(a);
                delete this.nh[a];
                delete this.mh[a]
            }
        };
    var E_a = class {
        constructor(a) {
            this.mh = a;
            _.Hn(a, "set_at", b => {
                _.Wn(this, "set_at", b);
                const c = a.getLength();
                b === 0 && c > 1 && _.Wn(this, "set_at", c)
            });
            _.Hn(a, "insert_at", b => {
                _.Wn(this, "insert_at", b);
                const c = a.getLength();
                c === 2 ? _.Wn(this, "insert_at", 2) : b === 0 && c > 1 && _.Wn(this, "set_at", c)
            });
            _.Hn(a, "remove_at", b => {
                _.Wn(this, "remove_at", b);
                const c = a.getLength();
                c === 1 ? _.Wn(this, "remove_at", 1) : b === 0 && c > 1 && _.Wn(this, "set_at", c)
            })
        }
        getLength() {
            let a = this.mh.getLength();
            a > 1 && ++a;
            return a
        }
        getAt(a) {
            this.mh.getLength() === a &&
                a > 1 && (a = 0);
            return this.mh.getAt(a)
        }
    };
    var F_a = class extends _.$n {
        constructor(a, b, c) {
            super();
            this.mh = a;
            this.nh = b;
            this.interpolate = c;
            this.Wh = [_.Rn(a, "set_at", this, this.qh), _.Rn(a, "insert_at", this, this.oh), _.Rn(a, "remove_at", this, this.ph)];
            for (let d = 0, e = a.getLength(); d < e; ++d) this.oh(d)
        }
        release() {
            this.Wh.forEach(_.Jn)
        }
        geodesic_changed() {
            z_a(this)
        }
        projection_changed() {
            z_a(this)
        }
        qh(a) {
            A_a(this, !1, a)
        }
        oh(a) {
            A_a(this, !0, a)
        }
        ph(a) {
            const b = this.get("projection");
            if (b) {
                var c = this.mh,
                    d = this.nh,
                    e = this.get("geodesic");
                a < c.getLength() ? (a > 0 && d.setAt(a -
                    1, wT(c.getAt(a - 1), c.getAt(a), e, b, this.interpolate)), d.removeAt(a)) : a > 0 && d.removeAt(a - 1)
            }
        }
    };
    var C_a = class extends _.MB {
        constructor(a, b, c, d, e) {
            super(a, b);
            c !== void 0 && (this.path = c);
            d !== void 0 && (this.vertex = d);
            e !== void 0 && (this.edge = e)
        }
    };
    var J_a = class extends _.$n {
        panes_changed() {
            AT(this)
        }
        paths_changed() {
            AT(this)
        }
        suppressGhostControlPoints_changed() {
            AT(this)
        }
        constructor(a, b, c, d, e) {
            super();
            this.Qm = a;
            this.Gn = b;
            this.ph = c;
            this.interpolate = d;
            this.Mh = e;
            this.nh = [];
            this.mh = [];
            this.oh = []
        }
        map_changed() {
            this.get("map") || (this.Qm.set("map", null), B_a(this))
        }
    };
    var FT = class extends _.$n {
            constructor(a) {
                super();
                this.nh = this.ph = null;
                this.mh = document.createElement("button");
                this.mh.type = "button";
                this.mh.style.background = "transparent";
                this.mh.style.border = "none";
                this.mh.style.margin = "0";
                this.mh.style.padding = "0";
                this.mh.title = "Undo last edit";
                this.mh.setAttribute("aria-label", "Undo last edit");
                const b = document.createElement("span");
                b.style.display = "inline-block";
                this.mh.appendChild(b);
                _.ux(this.mh, -202);
                this.mh.style.cursor = "pointer";
                this.mh.style.display = "none";
                this.Ky = _.xL(h1a, b, _.cp, i1a);
                this.Ky.style.position = "relative";
                _.Rn(this, "toolbar", this, this.rh);
                this.qh = new _.lN(a.wk, {
                    kp: c => {
                        c && (c = new _.Jo(c.Sh, c.Th), _.sx(this.mh, new _.Jo(c.x + j1a.x, c.y + j1a.y)))
                    },
                    zt: () => {}
                }, null, a);
                a.Cj(this.qh);
                _.jy(this.mh, {
                    zl: c => {
                        _.Vx(c);
                        this.oh(k1a)
                    },
                    Or: c => {
                        _.Wx(c)
                    },
                    wm: c => {
                        _.Wx(c)
                    },
                    Ol: c => {
                        _.Xx(c)
                    },
                    nl: ({
                        event: c
                    }) => {
                        _.Zw(c.mh);
                        c = c.mh;
                        _.En(c);
                        _.Fn(c);
                        this.ph();
                        this.Pk()
                    },
                    vr: {
                        qn: (c, d) => {
                            _.EBa(d)
                        },
                        pn: () => {},
                        Ln: () => {}
                    }
                });
                new _.LB(this.mh, this.mh, {
                    tt: this.oh.bind(this, _.cp),
                    ut: this.oh.bind(this,
                        l1a)
                })
            }
            panes_changed() {
                const a = this.get("panes");
                a ? a.floatPane.appendChild(this.mh) : this.mh.parentNode && _.Dl(this.mh)
            }
            Pk() {
                this.mh.style.display = "none";
                this.ph = null;
                K_a(this);
                this.nh = null
            }
            rh(a) {
                a.show && a.poly ? a.poly.get("suppressUndo") ? this.Pk() : (a.Ex && this.qh.setPosition(a.Ex), L_a(this, a.action || null, a.poly)) : a.poly === this.nh && this.Pk()
            }
            oh(a) {
                _.yL(this.Ky, i1a, a)
            }
        },
        h1a = _.$r("undo_poly"),
        i1a = new _.No(30, 27),
        l1a = new _.Jo(30, 0),
        k1a = new _.Jo(60, 0),
        j1a = new _.Jo(10, -11);
    FT.prototype.changed = FT.prototype.Pk;
    var B0a = class {
            constructor(a, b, c, d) {
                this.nh = b;
                this.oh = d;
                this.mh = null;
                this.Nl = !1;
                this.zIndex = 30;
                this.ph = new _.mN(a.elements, c, d)
            }
            Ft(a, b) {
                if (a !== "dragstart" && a !== "drag" && a !== "dragend") return !0;
                a = (a = (b.mh || b).get("map")) && a.get("draggable") !== !1 && a.get("gestureHandling") !== "none";
                return !!this.mh || !a
            }
            Nt(a, b) {
                var c = this.oh.get();
                if (!c) return null;
                c = c.fi.size;
                const d = _.FM(this.ph, a.hj, m1a);
                if (!d) return null;
                const e = new _.Jo(d.ju.Zh * c.Sh, d.ju.ai * c.Th),
                    f = new _.Jo(d.Bo.Zh * c.Sh, d.Bo.ai * c.Th),
                    g = [];
                var h = d.fl.zj;
                for (var k of Object.values(h)) g.push(k);
                g.reverse();
                g.sort((r, t) => t.zIndex - r.zIndex || 0);
                k = null;
                h = b ? 15 : 0;
                for (let r = 0, t = g.length; r < t; ++r) {
                    var n = g[r];
                    const v = n.rC;
                    if (!v.clickable) continue;
                    const w = n.GG.KA;
                    var p = v.dL / 2 + h;
                    n = M_a(n.paths);
                    if (p = KZa(e.x, e.y, p, n)) {
                        k = w;
                        b = this.oh.get();
                        b = _.Oy(b && b.fi, {
                            Zh: (p[0] + f.x) / c.Sh,
                            ai: (p[1] + f.y) / c.Th,
                            ii: d.Bo.ii
                        });
                        a.hj = new _.Jo(b.mh, b.nh);
                        a.latLng = _.Ar(b);
                        break
                    }
                    if (v.Gn && !b && LZa(e.x, e.y, n)) {
                        k = w;
                        break
                    }
                }
                return k
            }
            handleEvent(a, b, c) {
                !this.mh || a !== "mousedown" && a !== "dragstart" &&
                    a !== "drag" || (this.Nl = !0);
                this.mh && !this.Nl && (b.hj = this.mh.hj, b.latLng = this.mh.latLng);
                var d = c.mh || c;
                !this.mh && a === "mousedown" && d.get("draggable") && (this.mh = b, c.set("capturing", !0));
                d = c;
                if (this.mh) {
                    if (this.Nl && a === "dragend" || !this.Nl && a === "mouseup") this.Nl = !1, this.mh = null, c.set("capturing", !1);
                    a !== "dragstart" && a !== "drag" && a !== "dragend" || !c.mh || (d = c.mh);
                    a === "dragstart" && (a = "movestart");
                    a === "drag" && (a = "move");
                    a === "dragend" && (a = "moveend")
                } else {
                    if (a === "dragstart" || a === "dragend") return;
                    a === "drag" && (a = "mousemove")
                }
                a ===
                    "mouseout" ? this.nh.set("cursor", "") : a === "mousemove" && (c = c.get && c.get("cursor"), c === void 0 && (c = "pointer"), this.nh.set("cursor", c));
                _.Wn(d, a, new C_a(b.latLng, b.domEvent))
            }
        },
        m1a = new _.Jo(.5, .5);
    var m0a = () => [],
        T_a = () => {};
    m0a = a => {
        const b = [],
            c = a.length - 2;
        let d;
        for (let k = 2; k < c; k <<= 1) {
            var e = a[0],
                f = a[1];
            const n = Math.ceil(c / (2 * k)),
                p = Array(n);
            let r = 0;
            for (let t = 0, v = a.length - 1 - k; t < v;) {
                t += k;
                var g = a[t],
                    h = a[t + 1];
                t += k;
                t > a.length - 2 && (t = a.length - 2);
                const w = a[t],
                    y = a[t + 1],
                    E = g - e,
                    H = h - f;
                e = w - e;
                const M = y - f,
                    I = E * e + H * M;
                f = e * e + M * M;
                I >= f ? (g = w - g, h = y - h, h = g * g + h * h) : I <= 0 ? h = E * E + H * H : (h = E * M - H * e, h *= h, h /= f + 1E-16);
                h = Math.sqrt(h);
                d && (h += Math.max(d[2 * r], d[2 * r + 1] || 0));
                p[r++] = h;
                e = w;
                f = y
            }
            r < n && (p[r] = d ? d[2 * r] : 0);
            d = p;
            b.push(p)
        }
        return b
    };
    T_a = (a, b, c, d) => {
        --b;
        if (!(b < 0)) {
            var e = a.length,
                f = 0;
            for (d(0); f < b;) {
                const g = 1 << e,
                    h = e ? a[e - 1][f / g] : 0;
                let k = f + g;
                k >= b && (k = b);
                e && c(f, k, h) ? --e : (d(k), f = k, f & g || ++e)
            }
        }
    };
    var n1a = class {
        constructor(a, b, c, d) {
            this.zoom = b;
            this.fi = d;
            d = _.BI(this.fi, {
                Zh: a.x,
                ai: a.y,
                ii: b
            }, 16 / d.size.Sh);
            d = _.sp(d.min.mh, d.min.nh, d.max.mh, d.max.nh);
            this.projection = _.zr(c);
            var e = d,
                f = new _.Oq(e.minX, e.minY);
            e = new _.Oq(e.maxX, e.maxY);
            d = _.Ar(f, c, !0);
            var g = _.Ar(e, c, !0);
            e = Math.min(d.lat(), g.lat());
            f = Math.max(d.lat(), g.lat());
            var h = Math.min(d.lng(), g.lng());
            g = Math.max(d.lng(), g.lng());
            d = (h + g) / 2;
            const k = _.tm(d, -180, 180);
            h += k - d;
            g += k - d;
            d = _.sp(e, h, f, g);
            e = new _.pn(e, h, !0);
            h = new _.pn(f, g, !0);
            f = _.Cx(e, c);
            f =
                new _.Jo(f.mh, f.nh);
            e = _.Cx(h, c);
            e = new _.rp([f, new _.Jo(e.mh, e.nh)]);
            this.mh = d;
            this.oh = e;
            this.Bo = a;
            a = 1 << b;
            b = this.projection;
            c = .5 / a;
            d = this.oh;
            this.ph = Math.min(SZa(b, c, new _.Jo(d.minX, d.minY)), SZa(b, c, new _.Jo(d.maxX, d.maxY)));
            this.qh = U_a(this.mh, this.ph, this.projection);
            this.nh = Q_a(a)
        }
        getBounds() {
            return this.mh
        }
    };
    var o1a = class extends _.mr {
        constructor(a, b, c) {
            super();
            this.tiles = a;
            this.ph = Math.min(2, _.Zr());
            this.qh = b;
            this.nh = c || _.XB
        }
        mh() {
            return {
                fi: this.nh,
                sm: 2,
                vl: this.oh.bind(this)
            }
        }
        oh(a, b = {}) {
            const c = document.createElement("div");
            var d = this.nh.size;
            c.style.width = `${d.Sh}px`;
            c.style.height = `${d.Th}px`;
            const e = new _.Jo(a.Zh, a.ai);
            d = this.get("projection");
            const f = new n1a(e, a.ii, d, this.nh),
                g = f.getBounds(),
                h = this.qh(this.ph, c);
            let k = !1,
                n;
            return {
                Pj: () => c,
                ln: () => k,
                loaded: new Promise(p => {
                    n = {
                        bounds: g,
                        ej: e,
                        div: c,
                        sC: h,
                        zj: {},
                        vm: () => {
                            k = !0;
                            p(void 0)
                        },
                        RN: f,
                        zoom: a.ii
                    };
                    _.rq(this.tiles, n)
                }),
                release: () => {
                    c.textContent = "";
                    h.release();
                    this.tiles.remove(n);
                    b.Zj && b.Zj()
                }
            }
        }
    };
    var f0a = class {
        constructor(a, b, c, d, e) {
            this.gu = a;
            this.container = b;
            this.oh = e;
            this.nh = this.mh = null;
            this.zj = {};
            this.Bm = 0;
            this.tileSize = c.size;
            this.ph = d
        }
        release() {
            d0a(this)
        }
    };
    var p1a = class {
        constructor(a, b) {
            this.fi = a;
            this.gu = b;
            this.mh = null
        }
    };
    var q1a = class {
        constructor(a, b, c, d) {
            this.nh = b;
            this.mh = d;
            a.mh = e => {
                i0a(this, e)
            };
            a.onRemove = e => {
                j0a(this, e)
            };
            c.mh = e => {
                k0a(this, e)
            };
            c.onRemove = e => {
                l0a(this, e)
            };
            Object.values(a.elements).forEach(e => {
                i0a(this, e)
            });
            Object.values(c.elements).forEach(e => {
                k0a(this, e)
            })
        }
    };
    var E0a = class {
        constructor(a, b, c) {
            this.poly = a;
            this.nh = c;
            this.oh = c.fromDivPixelToLatLng(b);
            this.mh = [];
            a.get("latLngs").forEach(d => {
                this.mh.push(d.getArray().slice(0))
            })
        }
        moveTo(a) {
            const b = this.poly.get("latLngs"),
                c = this.nh;
            a = c.fromDivPixelToLatLng(a);
            a = YS(c, a);
            const d = YS(c, this.oh),
                e = new _.Oq(d.mh - a.mh, d.nh - a.nh);
            this.mh.forEach((f, g) => {
                const h = b.getAt(g);
                f.forEach((k, n) => {
                    k = YS(c, k);
                    h.setAt(n, ZS(c, new _.Oq(k.mh - e.mh, k.nh - e.nh)))
                })
            })
        }
    };
    var v0a = new tT(1),
        DT = new tT(0),
        ET = new tT(0);
    var z0a = {
            value: 100,
            unit: "%"
        },
        A0a = {
            value: 0,
            unit: "px"
        };
    var f1a = class {
        constructor() {
            var a = _.vGa();
            this.oh = _.vHa;
            this.mh = _.wHa;
            this.nh = a
        }
    };
    var H0a = class {
        constructor(a) {
            this.map = a;
            this.Mh = this.nh = this.mh = this.oh = this.geometry = null;
            this.ui = a.__gm;
            const b = new _.jr;
            (new g1a(this.ui.uh, b)).bindTo("projection", a);
            var c = _.sp(-100, -200, 100, 200);
            const d = new _.iN(c);
            c = new _.iN(c);
            this.tiles = new _.jr;
            new q1a(b, d, this.tiles, c);
            const e = _.ox(a.getDiv());
            let f = void 0,
                g = null;
            this.ph = new _.Yo(null);
            _.GM(a, this.ph, "overlayLayer", 30);
            this.ui.nh.then(h => {
                this.Mh = h.Mh;
                G0a(this, h.Mh);
                _.jw(h.xs, k => {
                    if (k && f !== k.fi) {
                        g && g.unbindAll();
                        f = k.fi;
                        var n = new p1a(f, e);
                        g = new o1a(this.tiles, (...p) => g0a(n, ...p), f);
                        g.bindTo("projection", a);
                        this.ph.set(g.mh())
                    }
                })
            });
            _.Hn(this.ui.uh, "insert", h => {
                F0a(this, h)
            });
            _.Hn(this.ui.uh, "remove", h => {
                this.onRemove(h)
            })
        }
        onRemove(a) {
            if (this.mh && this.Mh) {
                oT(a);
                var b = this.mh;
                delete a.editable_changed;
                H_a(a);
                _.Wn(b, "toolbar", {
                    show: !1,
                    poly: a
                });
                a.th && (_.Jn(a.th), delete a.th);
                a.set("capturing", !1);
                delete a.capturing_changed
            }
        }
    };
    var V0a = class extends _.$n {
        constructor() {
            super();
            const a = new _.ru({
                clickable: !1,
                fillOpacity: 0
            });
            a.bindTo("map", this);
            a.bindTo("strokeColor", this);
            a.bindTo("strokeOpacity", this);
            a.bindTo("strokeWeight", this);
            const b = new _.PM(["bounds", "freeControlPoint", "freeVertexPosition"], "return", J0a);
            b.bindTo("bounds", this);
            b.bindTo("freeControlPoint", this);
            b.bindTo("freeVertexPosition", this);
            a.bindTo("bounds", b, "return");
            this.mh = _.QM();
            this.mh.bindTo("zIndex", this);
            a.bindTo("zIndex", this.mh, "ghostZIndex")
        }
    };
    var O0a = [12, 6, 9, 3],
        P0a = [1, 2, 4, 8],
        K0a = ["ne-resize", "nw-resize", "se-resize", "sw-resize"],
        M0a = ["row-resize", "col-resize"],
        L0a = [0],
        N0a = [2, 1],
        W0a = class extends _.$n {
            constructor(a, b, c) {
                super();
                this.mh = a;
                this.ph = this.rh = null;
                this.oh = !1;
                this.th = b;
                this.nh = new _.xp;
                _.Rn(this.nh, "set_at", this, this.uh);
                this.qh = new _.xp;
                _.Rn(this.qh, "set_at", this, this.wh);
                this.sh = c
            }
            map_changed() {
                this.get("map") || (this.mh.set("map", null), GT(this.rh), this.get("suppressGhostControlPoints") || GT(this.ph), _.Mn(this.nh))
            }
            bounds_changed() {
                var a =
                    this.nh;
                const b = this.qh,
                    c = this.get("bounds");
                if (c) {
                    this.oh = !0;
                    a.setAt(0, c.getNorthEast());
                    a.setAt(1, _.iw(c));
                    a.setAt(2, _.hw(c));
                    a.setAt(3, c.getSouthWest());
                    a = c.bj.center();
                    var d = c.xi.center();
                    b.setAt(0, new _.pn(c.bj.lo, d));
                    b.setAt(1, new _.pn(a, c.xi.lo));
                    b.setAt(2, new _.pn(c.bj.hi, d));
                    b.setAt(3, new _.pn(a, c.xi.hi));
                    this.oh = !1;
                    _.Wn(this, "toolbar", {
                        show: !1
                    })
                } else a.clear(), b.clear()
            }
            uh(a) {
                R0a(this, O0a[a], this.nh.getAt(a))
            }
            wh(a) {
                R0a(this, P0a[a], this.qh.getAt(a))
            }
            panes_changed() {
                GT(this.rh);
                GT(this.ph);
                this.ph = null;
                const a = this.get("panes");
                a && (this.rh = Q0a(this, a, !0), this.get("suppressGhostControlPoints") || (this.ph = Q0a(this, a, !1)))
            }
            suppressGhostControlPoints_changed() {
                this.panes_changed()
            }
        };
    var X0a = class {
        constructor(a, b, c) {
            this.qh = a;
            this.nh = c;
            a = a.get("bounds");
            this.ph = YS(c, a.getSouthWest());
            this.oh = YS(c, a.getNorthEast());
            this.mh = YS(c, c.fromDivPixelToLatLng(b))
        }
        moveTo(a) {
            var b = this.nh;
            a = b.fromDivPixelToLatLng(a);
            b = YS(b, a);
            a = new _.Oq(this.oh.mh - this.mh.mh + b.mh, this.oh.nh - this.mh.nh + b.nh);
            b = new _.wo(ZS(this.nh, new _.Oq(this.ph.mh - this.mh.mh + b.mh, this.ph.nh - this.mh.nh + b.nh)), ZS(this.nh, a));
            this.qh.set("bounds", b)
        }
    };
    var $0a = class extends _.$n {
        constructor(a) {
            super();
            const b = new _.fu;
            this.mh = b;
            b.mh = a;
            b.rh = !0;
            b.bindTo("capturing", this);
            b.bindTo("cursor", this);
            b.bindTo("map", this);
            b.bindTo("strokeColor", this);
            b.bindTo("strokeOpacity", this);
            b.bindTo("strokeWeight", this);
            b.bindTo("strokePosition", this);
            b.bindTo("fillColor", this);
            b.bindTo("fillOpacity", this);
            b.bindTo("clickable", this);
            b.bindTo("zIndex", this);
            b.bindTo("suppressUndo", this);
            const c = [];
            this.nh = c;
            sT.forEach(d => {
                c.push(_.Vn(b, d, this))
            });
            c.push(_.Vn(this,
                "toolbar", b))
        }
        bounds_changed() {
            const a = this.mh;
            if (a) {
                var b = this.get("bounds");
                if (b) {
                    const c = b.getSouthWest(),
                        d = b.getNorthEast();
                    b = b.getCenter();
                    a.setPaths([new _.pn(d.lat(), d.lng()), new _.pn(d.lat(), b.lng()), new _.pn(d.lat(), c.lng()), new _.pn(c.lat(), c.lng()), new _.pn(c.lat(), b.lng()), new _.pn(c.lat(), d.lng())])
                } else a.setPaths([])
            }
        }
        release() {
            const a = this.nh;
            for (let b = 0, c = a.length; b < c; ++b) _.Jn(a[b]);
            delete this.nh;
            this.mh.unbindAll();
            delete this.mh
        }
    };
    var a1a = class {
        constructor(a) {
            this.mh = a;
            this.Mh = this.nh = null;
            this.ui = this.mh.__gm;
            this.ui.nh.then(({
                Mh: b
            }) => {
                this.Mh = b;
                Z0a(this, b)
            });
            this.ui.yh.mh = b => {
                Y0a(this, b)
            };
            this.ui.yh.onRemove = b => {
                this.onRemove(b)
            }
        }
        onRemove(a) {
            if (this.Mh) {
                var b = a.rh;
                b.unbindAll();
                b.set("map", null);
                b.release();
                delete a.rh;
                a.Wh && a.Wh.forEach(_.Jn);
                delete a.Wh;
                oT(a);
                delete a.editable_changed;
                T0a(a)
            }
        }
    };
    var r1a = class {
        constructor() {
            this.bJ = n_a;
            this.fJ = I0a;
            this.gJ = b1a
        }
    };
    _.Sl("poly", new r1a);
});