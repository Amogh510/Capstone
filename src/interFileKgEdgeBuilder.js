const traverse = require('@babel/traverse').default;
const path = require('path');
const { parseFile } = require('./astParser');
const { resolveAlias } = require('./aliasResolver');

/**
 * Build inter-file KG edges using FDG imports and a second AST pass.
 * Detects when a component in file A renders a component imported from file B,
 * then emits an edge: Component(A) -> Component(B) with type 'componentRendersComponent'.
 *
 * @param {{nodes: object[], edges: object[]}} fdg
 * @param {Map<string, object[]>} kgNodesByFile
 * @returns {object[]} edges
 */
function buildInterFileKgEdges(fdg, kgNodesByFile) {
  const edges = [];

  // Map: fileId -> { componentsByName: Map<name, {id,name,exportType}>, components: [] }
  const fileToComponents = new Map();
  for (const [fileId, nodes] of kgNodesByFile.entries()) {
    const comps = nodes.filter((n) => n.type === 'Component');
    const byName = new Map();
    comps.forEach((c) => byName.set(c.name, c));
    fileToComponents.set(fileId, { components: comps, componentsByName: byName });
  }

  // Build a quick lookup of which files each file imports
  const fileImports = new Map(); // fileId -> Set<importedFileId>
  (fdg.edges || []).forEach((e) => {
    if (!fileImports.has(e.from)) fileImports.set(e.from, new Set());
    fileImports.get(e.from).add(e.to);
  });

  // For each file with imports, analyze its AST to map import specifiers to target files
  (fdg.nodes || []).forEach((fileNode) => {
    const fileId = fileNode.fileId;
    const importsTo = fileImports.get(fileId);
    if (!importsTo || importsTo.size === 0) return;

    const ast = parseFile(fileId);
    if (!ast) return;

    // Component names in this file
    const srcComps = (fileToComponents.get(fileId) || {}).components || [];
    const srcCompNames = new Set(srcComps.map((c) => c.name));
    const srcCompByName = new Map(srcComps.map((c) => [c.name, c]));

    // import map: localName -> { sourceFile, importKind, importedName? }
    const importMap = new Map();

    traverse(ast, {
      ImportDeclaration(p) {
        const importPath = p.node.source.value;
        const resolved = resolveAlias(importPath, path.dirname(fileId));
        if (!importsTo.has(resolved)) return;

        (p.node.specifiers || []).forEach((spec) => {
          if (spec.type === 'ImportDefaultSpecifier') {
            importMap.set(spec.local.name, { sourceFile: resolved, importKind: 'default' });
          } else if (spec.type === 'ImportSpecifier') {
            const importedName = spec.imported ? spec.imported.name : spec.local.name;
            importMap.set(spec.local.name, { sourceFile: resolved, importKind: 'named', importedName });
          } else if (spec.type === 'ImportNamespaceSpecifier') {
            // Namespace imports cannot directly appear as <ns.Component/>; skip for component mapping
          }
        });
      },
    });

    // Helper to find enclosing component name
    function findEnclosingComponentName(startPath) {
      const fn = startPath.findParent((pp) => pp.isFunctionDeclaration() || pp.isVariableDeclarator());
      if (!fn) return null;
      if (fn.isFunctionDeclaration() && fn.node.id && fn.node.id.name) {
        const nm = fn.node.id.name;
        return srcCompNames.has(nm) ? nm : null;
      }
      if (fn.isVariableDeclarator() && fn.node.id && fn.node.id.name) {
        const nm = fn.node.id.name;
        return srcCompNames.has(nm) ? nm : null;
      }
      return null;
    }

    traverse(ast, {
      JSXOpeningElement(p) {
        const nameNode = p.node.name;
        if (!nameNode || nameNode.type !== 'JSXIdentifier') return;
        const tagName = nameNode.name;
        // Consider only custom components starting with uppercase
        if (!tagName || tagName[0] !== tagName[0].toUpperCase()) return;

        const imp = importMap.get(tagName);
        if (!imp) return; // not imported (could be local component)

        const enclosingName = findEnclosingComponentName(p);
        if (!enclosingName) return;
        const sourceComponent = srcCompByName.get(enclosingName);
        if (!sourceComponent) return;

        const targetFile = imp.sourceFile;
        const targetCompsInfo = fileToComponents.get(targetFile);
        if (!targetCompsInfo) return;

        let targetComponent = null;
        if (imp.importKind === 'default') {
          targetComponent = targetCompsInfo.components.find((c) => c.exportType === 'default') || targetCompsInfo.components[0] || null;
        } else if (imp.importKind === 'named') {
          targetComponent = targetCompsInfo.componentsByName.get(imp.importedName) || null;
        }
        if (!targetComponent) return;

        edges.push({
          from: sourceComponent.id,
          to: targetComponent.id,
          type: 'componentRendersComponent',
          intraFile: false,
          viaFileImport: true,
          fromFile: fileId,
          toFile: targetFile,
          viaLocalName: tagName,
        });
      },
    });
  });

  return edges;
}

module.exports = { buildInterFileKgEdges };

