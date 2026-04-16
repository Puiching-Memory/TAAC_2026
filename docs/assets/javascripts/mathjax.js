window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
  chtml: {
    adaptiveCSS: false,
  },
  startup: {
    ready() {
      MathJax.startup.defaultReady()

      if (typeof document$ !== "undefined") {
        document$.subscribe(() => {
          MathJax.typesetClear()
          MathJax.typesetPromise()
        })
      }
    },
  },
}
