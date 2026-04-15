/**
 * ECharts loader for MkDocs Material with automatic dark / light theme support.
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
    return ""
  }

  function isDark() {
    return (
      (document.body && document.body.getAttribute("data-md-color-scheme") === "slate") ||
      document.documentElement.getAttribute("data-md-color-scheme") === "slate"
    )
  }

  function clone(o) { return JSON.parse(JSON.stringify(o)) }

  /* ---- theme tokens ---- */
  var T = {
    dark: {
      text: "#cdd6f4", subtext: "#a6adc8",
      axis: "#585b70", split: "#313244",
      tipBg: "#313244", tipBorder: "#45475a",
      pieBorder: "#1e1e2e",
    },
    light: {
      text: "#4c4f69", subtext: "#6c6f85",
      axis: "#bcc0cc", split: "#e6e9ef",
      tipBg: "#eff1f5", tipBorder: "#ccd0da",
      pieBorder: "#ffffff",
    },
  }

  function tok() { return isDark() ? T.dark : T.light }

  /* Forcefully inject theme colors into a cloned ECharts option */
  function applyTheme(opt) {
    var t = tok()

    /* global text */
    opt.textStyle = opt.textStyle || {}
    opt.textStyle.color = t.text

    /* title */
    if (opt.title) {
      opt.title.textStyle = opt.title.textStyle || {}
      opt.title.textStyle.color = t.text
    }

    /* legend */
    if (opt.legend) {
      opt.legend.textStyle = opt.legend.textStyle || {}
      opt.legend.textStyle.color = t.text
    }

    /* tooltip */
    if (opt.tooltip) {
      opt.tooltip.backgroundColor = t.tipBg
      opt.tooltip.borderColor = t.tipBorder
      opt.tooltip.textStyle = opt.tooltip.textStyle || {}
      opt.tooltip.textStyle.color = t.text
    }

    /* xAxis / yAxis – support both object and array forms */
    function themeAxis(ax) {
      if (!ax) return
      var axes = Array.isArray(ax) ? ax : [ax]
      for (var i = 0; i < axes.length; i++) {
        var a = axes[i]
        a.axisLabel = a.axisLabel || {}
        a.axisLabel.color = t.subtext
        a.axisLine = a.axisLine || {}
        a.axisLine.lineStyle = a.axisLine.lineStyle || {}
        a.axisLine.lineStyle.color = t.axis
        a.splitLine = a.splitLine || {}
        a.splitLine.lineStyle = a.splitLine.lineStyle || {}
        a.splitLine.lineStyle.color = t.split
        /* axis name */
        a.nameTextStyle = a.nameTextStyle || {}
        a.nameTextStyle.color = t.subtext
      }
    }
    themeAxis(opt.xAxis)
    themeAxis(opt.yAxis)

    /* visualMap */
    if (opt.visualMap) {
      var vm = Array.isArray(opt.visualMap) ? opt.visualMap : [opt.visualMap]
      for (var i = 0; i < vm.length; i++) {
        vm[i].textStyle = vm[i].textStyle || {}
        vm[i].textStyle.color = t.text
      }
    }

    /* radar */
    if (opt.radar) {
      var rd = Array.isArray(opt.radar) ? opt.radar : [opt.radar]
      for (var i = 0; i < rd.length; i++) {
        rd[i].axisName = rd[i].axisName || {}
        rd[i].axisName.color = t.subtext
        rd[i].splitLine = rd[i].splitLine || {}
        rd[i].splitLine.lineStyle = rd[i].splitLine.lineStyle || {}
        rd[i].splitLine.lineStyle.color = t.split
        rd[i].splitArea = rd[i].splitArea || {}
        rd[i].splitArea.areaStyle = rd[i].splitArea.areaStyle || {}
        rd[i].splitArea.areaStyle.color = ["transparent", "transparent"]
      }
    }

    /* series: pie borderColor, label colors */
    if (opt.series) {
      for (var i = 0; i < opt.series.length; i++) {
        var s = opt.series[i]
        /* pie slice border = page background */
        if (s.type === "pie" && s.itemStyle && s.itemStyle.borderWidth) {
          s.itemStyle.borderColor = t.pieBorder
        }
        /* ensure bar/line labels are readable */
        if (s.label && s.label.show !== false) {
          s.label.color = s.label.color || t.subtext
        }
      }
    }

    opt.backgroundColor = "transparent"
    return opt
  }

  /* ---- chart registry ---- */
  var charts = [] /* { el, rawOption, instance, ro } */

  function attachResize(entry) {
    if (entry.ro) entry.ro.disconnect()
    var ro = new ResizeObserver(function () {
      if (entry.instance && !entry.instance.isDisposed()) entry.instance.resize()
    })
    ro.observe(entry.el)
    entry.ro = ro
  }

  function renderChart(container, rawOption) {
    container.style.width = "100%"
    var height = rawOption._height || "400px"
    container.style.minHeight = height
    delete rawOption._height
    container.textContent = ""
    var instance = echarts.init(container, null, { renderer: "canvas" })
    var themed = applyTheme(clone(rawOption))
    instance.setOption(themed)
    var entry = { el: container, rawOption: rawOption, instance: instance, ro: null }
    charts.push(entry)
    attachResize(entry)
  }

  var _reThemeTimer = 0
  function reThemeAll() {
    clearTimeout(_reThemeTimer)
    _reThemeTimer = setTimeout(function () {
      console.log("[echarts-fence] reThemeAll – isDark:", isDark(), "charts:", charts.length)
      for (var i = 0; i < charts.length; i++) {
        try {
          var c = charts[i]
          if (c.ro) c.ro.disconnect()
          c.instance.dispose()
          c.el.textContent = ""
          var inst = echarts.init(c.el, null, { renderer: "canvas" })
          var themed = applyTheme(clone(c.rawOption))
          inst.setOption(themed)
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
      var url = src.startsWith("http") ? src : getBase() + "/" + src
      el.textContent = "Loading chart…"
      fetch(url)
        .then(function (r) {
          if (!r.ok) throw new Error(r.status + " " + r.statusText)
          return r.json()
        })
        .then(function (opt) { renderChart(el, opt) })
        .catch(function (e) { el.textContent = "ECharts load error: " + e.message })
    } else {
      var raw = el.textContent.trim()
      if (raw) {
        try { renderChart(el, JSON.parse(raw)) }
        catch (e) { el.textContent = "ECharts JSON error: " + e.message }
      }
    }
  }

  /* ---- bootstrap ---- */
  function init() {
    if (typeof echarts === "undefined") { setTimeout(init, 100); return }
    for (var i = 0; i < charts.length; i++) {
      try {
        if (charts[i].ro) charts[i].ro.disconnect()
        charts[i].instance.dispose()
      } catch (_) {}
    }
    charts = []
    document.querySelectorAll("div.echarts").forEach(processDiv)
    console.log("[echarts-fence] init – loaded", charts.length, "charts, isDark:", isDark())
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(function () { init() })
  } else {
    document.addEventListener("DOMContentLoaded", init)
  }

  /* ---- theme toggle watchers ---- */
  var obs = new MutationObserver(function () { reThemeAll() })
  function watch(target) {
    if (target) obs.observe(target, { attributes: true, attributeFilter: ["data-md-color-scheme"] })
  }
  watch(document.documentElement)
  if (document.body) { watch(document.body) }
  else { document.addEventListener("DOMContentLoaded", function () { watch(document.body) }) }

  /* fallback: palette radio change */
  document.addEventListener("change", function (e) {
    var t = e.target
    if (t && t.getAttribute && t.getAttribute("name") === "__palette") {
      setTimeout(reThemeAll, 50)
    }
  })
})()
