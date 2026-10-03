# Feiyang's Log

A minimal, long-form research blog on robotics, embodied AI, world models, and machine learning. It's built with [Hugo](https://gohugo.io/) and a small custom theme that lives directly in this repo, with no external theme or CMS.

## Requirements

- **Hugo extended ≥ 0.132** (developed and deployed with **0.136.5**; also builds on 0.167).
  Math rendering uses Hugo's built-in `transform.ToMath` (KaTeX), which needs 0.132+.
  - macOS: `brew install hugo`
  - Other platforms: <https://gohugo.io/installation/>
- Nothing else: no Node, npm, or Go modules.

## Local development

```bash
hugo server            # http://localhost:1313, live reload
hugo server -D         # also show posts with `draft: true`
```

Production build (output goes to `public/`):

```bash
hugo --gc --minify
```

## Writing a post

```bash
hugo new content posts/my-post.md
```

This creates `content/posts/my-post.md` from `archetypes/posts.md` with `draft: true`. Remove that line (or set `draft: false`) to publish.

Front matter:

```yaml
---
title: "Latent Actions as Compact Futures"
date: 2026-10-03
tags: ["latent-actions", "world-models"]
summary: "Optional. Shown on the homepage. Defaults to the first paragraph."
# subtitle: "Optional line under the title"
# notice: "Optional small note above the post (Markdown)"
# showToc: false        # hide the table of contents
# showCitation: false   # hide the "Cited as / BibTeX" block
# draft: true
---
```

Post URLs look like `/posts/2026-10-03-my-post/` (set under `[permalinks]` in `hugo.toml`). Put `<!--more-->` after the opening paragraph to control where the auto-summary ends.

### Supported Markdown

Headings, lists, tables, blockquotes, footnotes (`[^1]`), fenced code blocks with syntax highlighting, and raw HTML all work.

**Math** (KaTeX, rendered at build time so the page ships no math JS):

```markdown
Inline: $z_t = E(o_t, o_{t+1})$  or  \(z_t\)

$$
\mathcal{L}
=
\|D(o_t, z_t) - o_{t+1}\|_2^2
$$
```

Display math may span several lines. To write a literal dollar sign, escape it as `\$`.

**Links:**

- External: `[Bruce et al., 2024](https://arxiv.org/abs/2402.15391)`. External links open in a new tab.
- Internal: `[another post]({{< relref "what-should-a-world-model-predict" >}})`

## Images & figures

Put images in `static/images/` (e.g. `static/images/my-post/overview.png`) and refer to them as `/images/my-post/overview.png`.

A stand-alone Markdown image becomes a centered, numbered figure. The caption comes from the title if one is given, otherwise from the alt text:

```markdown
![Overview of the proposed framework.](/images/my-post/overview.png)
```

For more control, use the `figure` shortcode. `caption` accepts Markdown and inline `$math$`:

```markdown
{{< figure src="/images/my-post/overview.png" caption="Overview of the proposed framework." width="80%" >}}
```

For architecture diagrams, `widefigure` extends past the text column, up to 960px:

```markdown
{{< widefigure src="/images/my-post/architecture.svg" caption="Full model architecture." >}}
```

Both shortcodes also accept `alt`, `link`, and `class`. Figures are numbered automatically ("Fig. 1.", "Fig. 2.", …).

*Page bundles also work:* create `content/posts/my-post/index.md` and put images next to it, then write `src="overview.png"`.

## Configuration

Everything you'll likely edit is in **`hugo.toml`**:

| What | Where |
|---|---|
| Site title, author | `title`, `params.author` |
| Description (meta / RSS) | `params.description` |
| Homepage "Welcome" intro | `[params.home]` |
| About-page links (GitHub, Scholar, Email…) | `[[params.links]]` (empty `url` hides a link) |
| Navigation | `[[menus.main]]` |
| TOC / citation / reading-time defaults | `params.showToc`, `params.showCitation`, `params.showReadingTime` |
| Deployed URL | `baseURL` (see Deployment) |

The About page text is in `content/about.md`.

Design tokens (colors, fonts, widths) are at the top of `assets/css/main.css`.

## Project layout

```
hugo.toml                 site configuration
content/
  posts/                  blog posts (Markdown)
  about.md archive.md search.md
layouts/                  the theme (templates)
  _default/               baseof, single, list, archive, search, terms, term
  _default/_markup/       render hooks: math, images, headings, links
  partials/               header, footer, post entry, TOC, citation, …
  shortcodes/             figure, widefigure
  index.html, index.json  homepage + search index
assets/css/               main.css (design), syntax.css (code colors)
assets/js/                theme toggle, search
static/                   favicon, images
.github/workflows/hugo.yml  GitHub Pages deployment
```

## Deployment (GitHub Pages)

1. Push this repo to GitHub.
2. Under **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**.
3. Push to `main`. `.github/workflows/hugo.yml` builds the site with Hugo 0.136.5 and deploys it.

The workflow passes the correct `baseURL` from your Pages settings, so this works for both
`https://<username>.github.io/` (repo named `<username>.github.io`) and `https://<username>.github.io/<repo>/`.
Still, set `baseURL` in `hugo.toml` to the final URL (see the TODO there) so local production builds match.
You never need to commit `public/`.

To upgrade Hugo in CI, change `HUGO_VERSION` in the workflow.

## Demo content

The three posts in `content/posts/` are placeholders for checking the layout (math, figures, tables, code). Delete them, along with `static/images/demo/`, once you have real posts.
