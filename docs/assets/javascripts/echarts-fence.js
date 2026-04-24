/**
 * ECharts loader for Zensical (Material-themed) docs with automatic dark / light
 * theme support.  Depends on ``document$`` and other Material-style globals.
 *
 * Usage in Markdown:
 *   <div class="echarts" data-src="assets/figures/eda/foo.echarts.json"></div>
 *   <div class="echarts">{ inline JSON }</div>
 */
;(function () {
  "use strict"

  /* ---- helpers ---- */
  function getBase() {
    if (typeof __md_scope !== "undefined")
      return __md_scope.href.replace(/\/$/, "")
    if (document.baseURI)
      return document.baseURI.replace(/\/$/, "")
    return window.location.href.replace(/\/$/, "")
  }

  function isDark() {
    return (
      (document.body && document.body.getAttribute("data-md-color-scheme") === "slate") ||
      document.documentElement.getAttribute("data-md-color-scheme") === "slate"
    )
  }

  function clone(o) { return JSON.parse(JSON.stringify(o)) }

  function resolveChartUrl(src) {
    if (!src) return ""
    if (/^https?:\/\//.test(src)) return src
    if (src.charAt(0) === "/") {
      return new URL(src, window.location.origin).href
    }

    // Zensical may rewrite raw HTML asset paths to ../assets/... for nested pages.
    // Treat docs asset references as site-rooted so repo subpaths remain intact.
    if (/^(?:\.\.\/)*assets\//.test(src)) {
      var siteRelative = src.replace(/^(?:\.\.\/)+(?=assets\/)/, "")
      return new URL(siteRelative, getBase() + "/").href
    }

    return new URL(src, window.location.href).href
  }

  /* Only the document dark/light mode is handled here.
     Visual colors should come from the chart option or ECharts theme itself. */
  var TT = {
    dark: {
      text: "#cdd6f4",
      subtext: "#a6adc8",
      tipBorder: "#45475a",
      link: "#94e2d5",
    },
    light: {
      text: "#4c4f69",
      subtext: "#6c6f85",
      tipBorder: "#ccd0da",
      link: "#176b5e",
    },
  }

  function tooltipTokens() { return isDark() ? TT.dark : TT.light }

  function chartThemeName() {
    return isDark() ? "dark" : null
  }

  function hasTopLegend(opt) {
    var legends = opt && opt.legend
    if (!legends) return false
    if (!Array.isArray(legends)) legends = [legends]
    for (var i = 0; i < legends.length; i++) {
      var legend = legends[i]
      if (!legend) continue
      if (legend.top != null && legend.top !== "auto" && legend.top !== "bottom") {
        return true
      }
    }
    return false
  }

  function defaultToolboxTop(opt) {
    var top = 8
    if (opt && opt.title) top += 34
    if (hasTopLegend(opt)) top += 28
    return top
  }

  function ensureFeature(featureMap, key, defaults) {
    var feature = featureMap[key]
    if (feature === false) return
    if (feature == null || feature === true) {
      featureMap[key] = defaults
      return
    }
    for (var name in defaults) {
      if (feature[name] == null) {
        feature[name] = defaults[name]
      }
    }
  }

  function ensureToolbox(opt) {
    var toolbox = opt.toolbox || {}
    toolbox.show = true
    if (toolbox.orient == null) toolbox.orient = "vertical"
    if (toolbox.right == null) toolbox.right = 8
    if (toolbox.top == null) toolbox.top = defaultToolboxTop(opt)
    if (toolbox.itemSize == null) toolbox.itemSize = 14
    if (toolbox.itemGap == null) toolbox.itemGap = 10
    toolbox.feature = toolbox.feature || {}

    ensureFeature(toolbox.feature, "saveAsImage", {
      title: "保存图片",
      pixelRatio: 2,
      excludeComponents: ["toolbox"],
    })
    ensureFeature(toolbox.feature, "dataView", {
      title: "数据视图",
      readOnly: true,
      lang: ["数据视图", "关闭", "刷新"],
    })
    ensureFeature(toolbox.feature, "restore", {
      title: "重置",
    })

    opt.toolbox = toolbox
  }

  function prepareOption(rawOption) {
    var option = clone(rawOption)
    if (option.backgroundColor == null) {
      option.backgroundColor = "transparent"
    }
    ensureToolbox(option)
    injectGraphTooltip(option)
    return option
  }

  /* ---- chart registry ---- */
  var charts = [] /* { el, rawOption, instance, ro } */

  /* Single global resize fallback for browsers without ResizeObserver */
  var _globalResizeBound = false
  function _ensureGlobalResize() {
    if (_globalResizeBound) return
    _globalResizeBound = true
    window.addEventListener("resize", function () {
      for (var i = 0; i < charts.length; i++) {
        try {
          if (charts[i].instance && !charts[i].instance.isDisposed()) {
            charts[i].instance.resize()
          }
        } catch (_) {}
      }
    })
  }

  function attachResize(entry) {
    if (entry.ro) entry.ro.disconnect()
    if (typeof ResizeObserver !== "undefined") {
      var ro = new ResizeObserver(function () {
        if (entry.instance && !entry.instance.isDisposed()) entry.instance.resize()
      })
      ro.observe(entry.el)
      entry.ro = ro
    } else {
      _ensureGlobalResize()
    }
  }

  /* ---- graph tooltip injection ---- */
  function hasGraphSeries(opt) {
    if (!opt || !opt.series) return false
    for (var i = 0; i < opt.series.length; i++) {
      if (opt.series[i] && opt.series[i].type === "graph") return true
    }
    return false
  }

  function updateContainerChrome(el, opt) {
    if (!el || !el.classList) return
    el.classList.toggle("echarts--graph-frame", hasGraphSeries(opt))
  }

  /**
   * For graph-type series whose nodes carry paper metadata (title, authors,
   * venue, paperYear, citations, abstract, paperUrl), inject a rich HTML
   * tooltip formatter.  This must run *after* JSON parsing because
   * JSON.parse strips functions.
   */
  function injectGraphTooltip(opt) {
    if (!opt || !opt.series) return
    for (var i = 0; i < opt.series.length; i++) {
      var s = opt.series[i]
      if (s.type !== "graph") continue
      /* Only inject if any node carries the "title" metadata key */
      var hasMetadata = s.data && s.data.some(function (d) { return !!d.title })
      if (!hasMetadata) continue

      opt.tooltip = opt.tooltip || {}
      opt.tooltip.trigger = "item"
      opt.tooltip.enterable = true
      opt.tooltip.confine = true
      opt.tooltip.padding = 0
      opt.tooltip.extraCssText = [
        "max-width:min(420px, calc(100vw - 32px))",
        "max-height:min(70vh, 420px)",
        "overflow:auto",
        "white-space:normal",
        "box-sizing:border-box",
        "padding:0",
        "border-radius:10px",
        "overflow-wrap:anywhere",
        "word-break:break-word",
      ].join(";")
      opt.tooltip.position = function (point, params, dom, rect, size) {
        var gap = 12
        var contentSize = (size && size.contentSize) || [
          (dom && dom.offsetWidth) || 0,
          (dom && dom.offsetHeight) || 0,
        ]
        var viewSize = (size && size.viewSize) || [
          window.innerWidth || 0,
          window.innerHeight || 0,
        ]
        var width = Math.min(contentSize[0], Math.max(viewSize[0] - gap * 2, 0))
        var height = Math.min(contentSize[1], Math.max(viewSize[1] - gap * 2, 0))
        var x = point[0] + gap
        if (x + width + gap > viewSize[0]) {
          x = point[0] - width - gap
        }
        x = Math.max(gap, Math.min(x, viewSize[0] - width - gap))

        var y = point[1] - Math.round(height / 2)
        y = Math.max(gap, Math.min(y, viewSize[1] - height - gap))
        return [x, y]
      }
      opt.tooltip.formatter = function (p) {
        var t = tooltipTokens()
        var cardWidth = Math.min(420, Math.max((window.innerWidth || 420) - 32, 240))
        if (p.dataType === "edge") {
          return _esc(p.data.sourceName || p.data.source) + " → " + _esc(p.data.targetName || p.data.target)
        }
        var d = p.data || {}
        var lines = []
        lines.push(
          "<div style='font-size:15px;font-weight:700;letter-spacing:0.01em;color:" + t.text + "'>"
          + _esc(d.name || p.name)
          + "</div>"
        )
        if (d.title && d.title !== d.name) {
          lines.push(
            "<div style='margin-top:4px;font-size:12px;line-height:1.55;color:" + t.subtext + "'>"
            + _esc(d.title)
            + "</div>"
          )
        }
        if (d.authors) {
          lines.push(
            "<div style='margin-top:10px'>"
            + "<div style='font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:" + t.subtext + "'>Authors</div>"
            + "<div style='margin-top:2px;font-size:12px;line-height:1.6;color:" + t.text + "'>" + _esc(d.authors) + "</div>"
            + "</div>"
          )
        }
        var meta = []
        if (d.paperYear) meta.push(String(d.paperYear))
        if (d.venue) meta.push(_esc(d.venue))
        if (d.citations != null) meta.push(d.citations.toLocaleString() + " citations")
        if (meta.length) {
          lines.push(
            "<div style='margin-top:10px'>"
            + "<div style='font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:" + t.subtext + "'>Publication</div>"
            + "<div style='margin-top:2px;font-size:12px;line-height:1.6;color:" + t.text + "'>" + meta.join(" · ") + "</div>"
            + "</div>"
          )
        }
        if (d.abstract) {
          lines.push(
            "<div style='margin-top:10px;padding-top:10px;border-top:1px solid " + t.tipBorder + "'>"
            + "<div style='font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:" + t.subtext + "'>Abstract</div>"
            + "<div style='margin-top:4px;font-size:12px;line-height:1.6;color:" + t.text + "'>"
            + _esc(d.abstract)
            + "</div>"
            + "</div>"
          )
        }
        var safePaperUrl = _safeExternalUrl(d.paperUrl)
        if (safePaperUrl) {
          lines.push(
            "<div style='margin-top:10px;padding-top:10px;border-top:1px solid " + t.tipBorder + "'>"
            + "<a href=\"" + _escAttr(safePaperUrl) + "\" target=\"_blank\" rel=\"noopener noreferrer\" style='font-size:12px;font-weight:600;letter-spacing:0.02em;color:" + t.link + ";text-decoration:none'>"
            + "查看论文原文"
            + "</a>"
            + "</div>"
          )
        }
        return "<div style='width:min(100%, " + cardWidth + "px);padding:12px 14px;line-height:1.6'>" + lines.join("") + "</div>"
      }
    }
  }

  function _esc(s) {
    if (!s) return ""
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  }

  function _escAttr(s) {
    if (!s) return ""
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;")
  }

  function _safeExternalUrl(s) {
    if (!s) return ""
    try {
      var url = new URL(String(s), document.baseURI)
      if (url.protocol === "http:" || url.protocol === "https:") {
        return url.href
      }
    } catch (e) {
      return ""
    }
    return ""
  }

  function renderChart(container, rawOption) {
    container.style.width = "100%"
    var height = rawOption._height || "400px"
    container.style.minHeight = height
    delete rawOption._height
    container.textContent = ""
    updateContainerChrome(container, rawOption)

    var instance = echarts.init(container, chartThemeName(), { renderer: "canvas" })
    instance.setOption(prepareOption(rawOption))
    var entry = { el: container, rawOption: rawOption, instance: instance, ro: null }
    charts.push(entry)
    attachResize(entry)
  }

  var _reThemeTimer = 0
  function reThemeAll() {
    clearTimeout(_reThemeTimer)
    _reThemeTimer = setTimeout(function () {

      for (var i = 0; i < charts.length; i++) {
        try {
          var c = charts[i]
          if (c.ro) c.ro.disconnect()
          c.instance.dispose()
          c.el.textContent = ""
          var inst = echarts.init(c.el, chartThemeName(), { renderer: "canvas" })
          inst.setOption(prepareOption(c.rawOption))
          c.instance = inst
          attachResize(c)
        } catch (e) {
          console.error("[echarts-fence] reTheme error on chart", i, e)
        }
      }
    }, 30)
  }

  /* ---- data loading ---- */
  function processDiv(el) {
    var src = el.getAttribute("data-src")
    if (src) {
      var url = resolveChartUrl(src)
      el.textContent = "Loading chart…"
      fetch(url)
        .then(function (r) {
          if (!r.ok) throw new Error(r.status + " " + r.statusText)
          return r.json()
        })
        .then(function (opt) { renderChart(el, opt) })
        .catch(function (e) { el.textContent = "ECharts load error: " + e.message })
    } else {
      var cached = el.getAttribute("data-option")
      var raw = el.textContent.trim()
      var source = raw || cached
      if (source) {
        try {
          var opt = JSON.parse(source)
          if (raw) el.setAttribute("data-option", raw)
          renderChart(el, opt)
        }
        catch (e) { el.textContent = "ECharts JSON error: " + e.message }
      }
    }
  }

  /* ---- bootstrap ---- */
  var _initRetries = 0
  var _maxInitRetries = 10

  function init() {
    if (typeof echarts === "undefined") {
      _initRetries++
      if (_initRetries > _maxInitRetries) {
        document.querySelectorAll("div.echarts").forEach(function (el) {
          if (!el.textContent.trim()) {
            el.textContent = "ECharts library failed to load. Please check your network connection."
          }
        })
        return
      }
      var delay = Math.min(100 * Math.pow(2, _initRetries - 1), 3200)
      setTimeout(init, delay)
      return
    }
    _initRetries = 0
    for (var i = 0; i < charts.length; i++) {
      try {
        if (charts[i].ro) charts[i].ro.disconnect()
        charts[i].instance.dispose()
      } catch (_) {}
    }
    charts = []
    document.querySelectorAll("div.echarts").forEach(processDiv)

  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(function () { init() })
  } else {
    document.addEventListener("DOMContentLoaded", init)
  }

  /* ---- theme toggle watchers ---- */
  if (typeof MutationObserver !== "undefined") {
    var obs = new MutationObserver(function () { reThemeAll() })
    function watch(target) {
      if (target) obs.observe(target, { attributes: true, attributeFilter: ["data-md-color-scheme"] })
    }
    watch(document.documentElement)
    if (document.body) { watch(document.body) }
    else { document.addEventListener("DOMContentLoaded", function () { watch(document.body) }) }
  }

  /* fallback: palette radio change */
  document.addEventListener("change", function (e) {
    var t = e.target
    if (t && t.getAttribute && t.getAttribute("name") === "__palette") {
      setTimeout(reThemeAll, 50)
    }
  })
})()
