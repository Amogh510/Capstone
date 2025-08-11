const path = require('path');
const traverse = require('@babel/traverse').default;
const { parseFile } = require('./astParser');
const { createKgNodeId } = require('./utils/idUtils');
// no tag allowlist restriction for CSS linking; match on any JSX tag

/**
 * Extract class names from a JSX attribute value string or simple template.
 */
function splitClassNames(value) {
  if (!value || typeof value !== 'string') return [];
  return value
    .split(/\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/**
 * Build indices of CSS selectors per fileId: { classes, ids, tags }
 */
function buildCssIndices(kgNodesByFile) {
  const indices = new Map();
  for (const [fileId, nodes] of kgNodesByFile.entries()) {
    const classes = new Map();
    const ids = new Map();
    const tags = new Map();
    nodes.filter((n) => n && n.type === 'CSSSelector').forEach((sel) => {
      const text = sel.selectorText || '';
      if (sel.selectorKind === 'class' && text.startsWith('.')) {
        const key = text.slice(1);
        if (!classes.has(key)) classes.set(key, []);
        classes.get(key).push(sel);
      } else if (sel.selectorKind === 'id' && text.startsWith('#')) {
        const key = text.slice(1);
        if (!ids.has(key)) ids.set(key, []);
        ids.get(key).push(sel);
      } else if (sel.selectorKind === 'tag') {
        const key = text.toLowerCase();
        if (!tags.has(key)) tags.set(key, []);
        tags.get(key).push(sel);
      }
    });
    indices.set(fileId, { classes, ids, tags });
  }
  return indices;
}

/**
 * For a given JS file, discover style files to consider:
 * - Directly imported style files (fdg.nodes[].stylesImported)
 * - Plus: any global CSS files imported by other top-level files (heuristic: non-module css)
 */
function discoverCandidateStyleFiles(fdg, kgNodesByFile, fileId) {
  const node = (fdg.nodes || []).find((n) => n.fileId === fileId);
  if (!node) return new Set();
  const set = new Set(node.stylesImported || []);
  // Heuristic global CSS: any .css not ending with .module.css imported anywhere
  (fdg.nodes || []).forEach((n) => {
    (n.stylesImported || []).forEach((s) => {
      if (/\.css$/i.test(s) && !/\.module\.css$/i.test(s)) set.add(s);
    });
  });
  // Additional heuristic: include all non-module CSS files that exist in the KG (global cascade)
  for (const [fid, nodes] of kgNodesByFile.entries()) {
    const hasCss = nodes.some((nn) => nn.type === 'CSSSelector' && nn.isModule === false);
    if (hasCss) set.add(fid);
  }
  return set;
}

/**
 * Scan a file AST to map `className` and inline style usage on allowed tags.
 * Returns: { jsxPerComponent: Map<componentName, Array<{tag, classNames[], id?, styleObjString?}>> }
 */
function scanJsxUsage(ast) {
  const jsxPerComponent = new Map();
  const cssModuleLocals = new Set();

  function ensureComp(name) {
    if (!jsxPerComponent.has(name)) jsxPerComponent.set(name, []);
    return jsxPerComponent.get(name);
  }

  function getEnclosingComponentName(startPath) {
    const fn = startPath.findParent((pp) => pp.isFunctionDeclaration() || pp.isVariableDeclarator());
    if (!fn) return null;
    if (fn.isFunctionDeclaration() && fn.node.id && fn.node.id.name) return fn.node.id.name;
    if (fn.isVariableDeclarator() && fn.node.id && fn.node.id.name) return fn.node.id.name;
    return null;
  }

  traverse(ast, {
    ImportDeclaration(p) {
      const src = (p.node.source && p.node.source.value) || '';
      if (/\.module\.(css|scss|sass)$/i.test(src)) {
        (p.node.specifiers || []).forEach((spec) => {
          if (spec.type === 'ImportDefaultSpecifier' && spec.local && spec.local.name) {
            cssModuleLocals.add(spec.local.name);
          } else if (spec.type === 'ImportNamespaceSpecifier' && spec.local && spec.local.name) {
            cssModuleLocals.add(spec.local.name);
          }
        });
      }
    },
    JSXOpeningElement(p) {
      const nameNode = p.node.name;
      if (!nameNode || nameNode.type !== 'JSXIdentifier') return;
      const tag = nameNode.name;

      const compName = getEnclosingComponentName(p);
      if (!compName) return;

      let classNames = [];
      let idValue = null;
      let inlineStyleExpr = null;

      (p.get('attributes') || []).forEach((ap) => {
        const attr = ap.node;
        if (!attr || !attr.name || !attr.name.name) return;
        const attrName = attr.name.name;
        if (attrName === 'className') {
          if (attr.value && attr.value.type === 'StringLiteral') {
            classNames = splitClassNames(attr.value.value);
          } else if (ap.get('value') && ap.get('value').isJSXExpressionContainer()) {
            const exp = ap.get('value.expression');
            // Handle CSS Modules: styles.foo -> 'foo'
            if (exp.isMemberExpression()) {
              const obj = exp.get('object');
              const prop = exp.get('property');
              if (obj && obj.isIdentifier() && cssModuleLocals.has(obj.node.name)) {
                if (prop.isIdentifier()) classNames.push(prop.node.name);
                else if (prop.isStringLiteral()) classNames.push(prop.node.value);
              }
            } else if (exp.isTemplateLiteral()) {
              try {
                const text = exp.node.quasis.map((q) => q.value.cooked || '').join(' ');
                classNames = classNames.concat(splitClassNames(text));
              } catch (_) {}
            } else {
              // Fallback best-effort: split textual representation (captures Tailwind)
              try {
                const s = exp.toString();
                classNames = classNames.concat(splitClassNames(s.replace(/[`'"+{}().]/g, ' ')));
              } catch (_) {}
            }
          }
        } else if (attrName === 'id') {
          if (attr.value && attr.value.type === 'StringLiteral') idValue = attr.value.value;
          else if (ap.get('value') && ap.get('value').isJSXExpressionContainer()) idValue = ap.get('value.expression').toString();
        } else if (attrName === 'style') {
          if (ap.get('value') && ap.get('value').isJSXExpressionContainer()) {
            inlineStyleExpr = ap.get('value.expression').toString();
          }
        }
      });

      ensureComp(compName).push({ tag, classNames, idValue, inlineStyleExpr });
    },
  });

  return { jsxPerComponent };
}

/**
 * Create InlineStyle nodes and TailwindUtility nodes as KG nodes.
 */
function createStyleRelatedNodes(fileId, jsxPerComponent, kgNodesByFile) {
  const nodes = [];
  const addNode = (n) => nodes.push(n);

  for (const [compName, elems] of jsxPerComponent.entries()) {
    elems.forEach((e, idx) => {
      if (e.inlineStyleExpr) {
        addNode({
          id: createKgNodeId('InlineStyle', `${compName}:${idx}`, compName, fileId),
          type: 'InlineStyle',
          expression: e.inlineStyleExpr,
          component: compName,
          filePath: fileId,
        });
      }
      // Tailwind utilities: treat each class token as TailwindUtility if it looks utility-like
      (e.classNames || []).forEach((cn) => {
        if (/^[a-z0-9-:/\[\]!]+$/i.test(cn)) {
          addNode({
            id: createKgNodeId('TailwindUtility', cn, compName, fileId),
            type: 'TailwindUtility',
            name: cn,
            component: compName,
            filePath: fileId,
          });
        }
      });
    });
  }

  if (!kgNodesByFile.has(fileId)) kgNodesByFile.set(fileId, []);
  kgNodesByFile.get(fileId).push(...nodes);
  return nodes;
}

/**
 * Build CSS/Tailwind/InlineStyle edges.
 */
function buildCssLinks(fdg, kgNodesByFile) {
  const edges = [];
  const cssIndex = buildCssIndices(kgNodesByFile);

  for (const fileNode of (fdg.nodes || [])) {
    const fileId = fileNode.fileId;
    const ast = parseFile(fileId);
    if (!ast) continue;

    const { jsxPerComponent } = scanJsxUsage(ast);
    // Create InlineStyle and TailwindUtility nodes
    createStyleRelatedNodes(fileId, jsxPerComponent, kgNodesByFile);

    const candidateStyles = discoverCandidateStyleFiles(fdg, kgNodesByFile, fileId);
    const directlyImportedStyles = new Set(fileNode.stylesImported || []);
    const styleIndices = [];
    candidateStyles.forEach((s) => {
      const idx = cssIndex.get(s);
      if (idx) styleIndices.push(idx);
    });

    // Helper to add edge
    const add = (from, to, type, intraFile) => {
      if (!from || !to) return;
      edges.push({ from, to, type, intraFile });
    };

    // Map component name to its KG id if present
    const fileKgNodes = kgNodesByFile.get(fileId) || [];
    const compByName = new Map();
    fileKgNodes.filter((n) => n.type === 'Component').forEach((c) => compByName.set(c.name, c.id));
    const jsxByCompId = new Map();

    for (const [compName, elems] of jsxPerComponent.entries()) {
      const compId = compByName.get(compName) || compName;
      elems.forEach((e, idx) => {
        // Create JSXElement lookup id to link edges to KG JSX nodes when available
        // We will try to find matching JSXElement node by tag and id/class if present
        const jsxNodes = fileKgNodes.filter((n) => n.type === 'JSXElement' && n.tagName === e.tag);
        // Class-based matches
        (e.classNames || []).forEach((cn) => {
          // Tailwind utility edge from component to utility node
          const twNodeId = createKgNodeId('TailwindUtility', cn, compName, fileId);
          add(compId, twNodeId, 'componentUsesTailwindUtility', true);

          // Match CSS class selectors from candidate style files
          styleIndices.forEach((idx) => {
            const list = (idx.classes.get(cn) || []);
            list.forEach((sel) => {
              add(compId, sel.id, 'componentUsesCssSelector', false);
              // Also link JSXElement to CSS selector when we can
              jsxNodes.forEach((j) => add(j.id, sel.id, 'jsxHasClass', false));
            });
          });
        });

        // ID-based matches
        if (e.idValue) {
          styleIndices.forEach((idx) => {
            const list = (idx.ids.get(e.idValue) || []);
            list.forEach((sel) => {
              add(compId, sel.id, 'componentUsesCssSelector', false);
              jsxNodes.forEach((j) => add(j.id, sel.id, 'jsxHasId', false));
            });
          });
        }

        // Tag-based matches (weak)
        styleIndices.forEach((idx) => {
          const list = (idx.tags.get(e.tag.toLowerCase()) || []);
          list.forEach((sel) => {
            add(compId, sel.id, 'componentMatchesTagSelector', false);
            jsxNodes.forEach((j) => add(j.id, sel.id, 'jsxMatchesTagSelector', false));
          });
        });

        // Inline style edges
        if (e.inlineStyleExpr) {
          const inlineNodeId = createKgNodeId('InlineStyle', `${compName}:${idx}`, compName, fileId);
          add(compId, inlineNodeId, 'componentHasInlineStyle', true);
          jsxNodes.forEach((j) => add(j.id, inlineNodeId, 'jsxHasInlineStyle', true));
        }
      });

      // Fallback linkage: if a stylesheet is directly imported by this file,
      // connect the component to all selectors from that stylesheet so the
      // selectors are not floating in the KG. This is weaker than explicit
      // usage edges and marked with a distinct type.
      directlyImportedStyles.forEach((styleFile) => {
        const styleNodes = (kgNodesByFile.get(styleFile) || []).filter((n) => n.type === 'CSSSelector');
        styleNodes.forEach((sel) => add(compId, sel.id, 'componentImportsStylesheetSelector', false));
      });
    }
  }

  return { edges };
}

module.exports = { buildCssLinks };

