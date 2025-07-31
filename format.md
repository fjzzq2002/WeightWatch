# WeightWatch Website Style Guide

This document outlines the key styling choices and components needed to maintain consistency across WeightWatch website pages.

## Core Dependencies

### CSS & JavaScript Libraries
```html
<!-- Main stylesheet -->
<link rel="stylesheet" type="text/css" media="all" href="assets/stylesheets/main_free.css" />

<!-- Code highlighting -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/styles/foundation.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/highlight.min.js"></script>
<script>hljs.initHighlightingOnLoad();</script>

<!-- Icons -->
<link href="assets/fontawesome-free-6.6.0-web/css/all.min.css" rel="stylesheet">

<!-- MathJax for mathematical notation -->
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        "HTML-CSS": {
          scale: 95,
          fonts: ["Gyre-Pagella"],
          imageFont: null,
          undefinedFamily: "'Arial Unicode MS', cmbright"
        },
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            processEscapes: true
          }
      });
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
```

## Typography & Fonts

### Primary Font Stack
- **Mathematical text**: Gyre-Pagella (via MathJax)
- **Fallbacks**: Arial Unicode MS, cmbright
- **Scale**: 95% for MathJax rendering

### Text Classes
```html
<p class="text">Regular body text</p>
<p class="caption">Figure captions</p>
<h1 class="title">Main page title</h1>
<p class="author">Author information</p>
<p class="abstract">Abstract text</p>
```

## Layout Structure

### Container System
```html
<!-- Title section with colored background -->
<div class="container blog" id="first-content" style="background-color: #E0E4E6;">
    <div class="blog-title no-cover">
        <!-- Title content -->
    </div>
</div>

<!-- Regular white background sections -->
<div class="container blog main">
    <!-- Content -->
</div>

<!-- Gray background sections -->
<div class="container blog main gray">
    <!-- Content (used for figures, alternating sections) -->
</div>

<!-- First main section -->
<div class="container blog main first" id="blog-main">
    <!-- Content -->
</div>
```

## Color Scheme

### Background Colors
- **Title section**: `#E0E4E6` (light blue-gray)
- **Regular sections**: White (default)
- **Alternate sections**: Gray (via `.gray` class)

### Button Styling
```html
<a href="#" class="button icon" style="background-color: rgba(255, 255, 255, 0.2)">
    Button Text <i class="fa-solid fa-icon"></i>
</a>
```

## Mathematical Notation

### LaTeX-style Bold Headers
```html
<!-- Use in results lists for consistent academic formatting -->
<p class="text">${\bf Section~Name}$ Regular text content...</p>
```

### Inline Math
- Use `$...$` for inline mathematical expressions
- Use `$$...$$` for display math blocks

## Code Blocks

### Syntax Highlighting
```html
<pre><code class="python">
# Python code here
def example():
    pass
</code></pre>

<pre><code class="plaintext">
Plain text code blocks (for citations, etc.)
</code></pre>
```

## Lists & Results

### Clean Results Lists
```html
<ul class="results-list">
    <li><p class="text">${\bf Item~Name}$ Description text...</p></li>
</ul>
```

### Custom List Styling
```css
.results-list {
    list-style: none;
    padding-left: 0;
}

.results-list li {
    margin-bottom: 0.5rem;
}
```

## Images & Figures

### Figure Layout
```html
<div class="container blog main gray">
    <img src="figure.png">
    <p class="caption">
        <strong>Figure X:</strong> Caption text with <em>emphasis</em> where needed.
    </p>
</div>
```

### Centered Figures
```html
<div style="width: 80%; margin: 0 auto;">
    <img src="figure.png">
    <p class="caption">Caption text</p>
</div>
```

## Footer

### Standard Footer
```html
<footer>
    <div class="container">
        <p style="text-align: center;">
            This website is built on the <a href="https://shikun.io/projects/clarity">Clarity Template</a>, designed by <a href="https://shikun.io/">Shikun Liu</a>.
        </p>
    </div>    
</footer>
```

## Meta Tags Template

### SEO & Social Media
```html
<meta name="description" content="Your page description">
<meta name="referrer" content="no-referrer-when-downgrade">
<meta name="robots" content="all">
<meta content="en_EN" property="og:locale">
<meta content="website" property="og:type">
<meta content="https://ziqianz.github.io/WeightWatch/page" property="og:url">
<meta content="Page Title" property="og:title">
<meta content="Page description" property="og:description">

<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@ziqian_zhong">
<meta name="twitter:description" content="Page description">
<meta name="twitter:image:src" content="image.png">
```

## Key Design Principles

1. **Alternating backgrounds**: Use white and gray sections for visual rhythm
2. **Mathematical consistency**: Use MathJax for all mathematical notation
3. **Clean typography**: Minimal styling, focus on readability
4. **Academic formatting**: LaTeX-style bold for section headers in results
5. **Responsive images**: Center important figures, use captions consistently
6. **Code highlighting**: Foundation theme for syntax highlighting
7. **Icon consistency**: Use FontAwesome for all icons