const fs = require('fs');
const path = require('path');
const postcss = require('postcss');
const scssSyntax = require('postcss-scss');
const selectorParser = require('postcss-selector-parser');
const specificity = require('specificity');
const { createKgNodeId } = require('../utils/idUtils');

function parseSpecificity(selector) {
  try {
    const res = specificity.calculate(selector);
    if (res && res.length > 0) {
      const s = res[0].specificity.split(',').map((n) => parseInt(n, 10));
      return { tuple: s, value: s[0] * 1000 + s[1] * 100 + s[2] * 10 + s[3] };
    }
  } catch (_) {}
  return { tuple: [0, 0, 0, 0], value: 0 };
}

function detectSelectorKind(selector) {
  let kind = 'complex';
  try {
    selectorParser((selectors) => {
      selectors.each((sel) => {
        // If the selector has more than one node, or contains combinators/pseudos, treat as complex
        const nodes = sel.nodes || [];
        const onlyOne = nodes.length === 1;
        if (onlyOne) {
          const n = nodes[0];
          if (n.type === 'class') kind = 'class';
          else if (n.type === 'id') kind = 'id';
          else if (n.type === 'tag') kind = 'tag';
          else kind = 'complex';
        } else {
          // But if nodes are like a single compound without combinators, still complex for v1
          kind = 'complex';
        }
      });
    }).processSync(selector);
  } catch (_) {}
  return kind;
}

function collectMediaQueries(rule) {
  const medias = [];
  let current = rule.parent;
  while (current) {
    if (current.type === 'atrule' && current.name === 'media') {
      medias.push(current.params || '');
    }
    current = current.parent;
  }
  return medias.reverse();
}

/**
 * Extract CSS selectors and declarations from a style file.
 * Returns array of KG nodes of type 'CSSSelector'.
 */
function extractCssSelectors({ filePath }) {
  try {
    const code = fs.readFileSync(filePath, 'utf-8');
    const ext = path.extname(filePath).toLowerCase();
    const isModule = /\.module\.(css|scss|sass)$/i.test(filePath);
    const root = postcss.parse(code, {
      from: filePath,
      syntax: ext === '.scss' || ext === '.sass' ? scssSyntax : undefined,
    });

    const selectors = [];

    root.walkRules((rule) => {
      const mediaQueries = collectMediaQueries(rule);
      const decls = {};
      rule.nodes && rule.nodes.forEach((n) => {
        if (n.type === 'decl' && n.prop) {
          decls[n.prop] = n.value;
        }
      });

      const selectorText = rule.selector || '';
      if (!selectorText) return;
      const parts = selectorText.split(',').map((s) => s.trim()).filter(Boolean);
      parts.forEach((sel) => {
        const spec = parseSpecificity(sel);
        const selectorKind = detectSelectorKind(sel);
        selectors.push({
          id: createKgNodeId('CSSSelector', sel, undefined, filePath),
          type: 'CSSSelector',
          selectorText: sel,
          selectorKind,
          specificity: spec.tuple,
          specificityValue: spec.value,
          declarations: decls,
          mediaQueries,
          filePath,
          isModule,
        });
      });
    });

    return selectors;
  } catch (err) {
    console.error('Error parsing CSS file', filePath, err.message);
    return [];
  }
}

module.exports = { extractCssSelectors };

