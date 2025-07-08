const path = require('path');
const fs = require('fs');
const traverse = require('@babel/traverse').default;
const { normalizePath } = require('./utils/fileUtils');
const { resolveAlias } = require('./aliasResolver');

/**
 * Builds the File Dependency Graph (FDG).
 * @param {string[]} files - An array of absolute file paths.
 * @param {Map<string, object[]>} kgNodesByFile - A map of KG nodes by file path.
 * @returns {{nodes: object[], edges: object[]}} The FDG.
 */
function buildFdg(files, kgNodesByFile) {
  const nodes = [];
  const edges = [];

  files.forEach((filePath) => {
    const fileId = normalizePath(filePath);
    const fileType = path.extname(filePath).substring(1);
    const isStyle = ['css', 'scss', 'sass'].includes(fileType);
    const linesOfCode = fs.readFileSync(filePath, 'utf-8').split('\n').length;
    const imports = [];
    const exports = [];
    const stylesImported = [];
    const kgNodeRefs = {};

    if (!isStyle) {
      const ast = require('./astParser').parseFile(filePath);
      if (ast) {
        traverse(ast, {
          ImportDeclaration(babelPath) {
            const importPath = babelPath.node.source.value;
            const resolvedPath = resolveAlias(
              importPath,
              path.dirname(filePath)
            );
            if (/\.(css|scss|sass)$/.test(resolvedPath)) {
              stylesImported.push(resolvedPath);
            } else {
              imports.push(resolvedPath);
              edges.push({
                from: fileId,
                to: resolvedPath,
                type: 'staticImport',
              });
            }
          },
          ExportNamedDeclaration(babelPath) {
            if (babelPath.node.declaration) {
              exports.push(babelPath.node.declaration.id.name);
            }
          },
          ExportDefaultDeclaration(babelPath) {
            exports.push('default');
          },
        });
      }
    }

    const fileKgNodes = kgNodesByFile.get(fileId) || [];
    fileKgNodes.forEach((node) => {
      if (!kgNodeRefs[node.type]) {
        kgNodeRefs[node.type] = [];
      }
      kgNodeRefs[node.type].push(node.id);
    });

    nodes.push({
      fileId,
      fileType,
      linesOfCode,
      imports,
      exports,
      stylesImported,
      kgNodeRefs,
      declareRoutes: false, // This will be updated by the routeExtractor
      isStyle,
    });
  });

  return { nodes, edges };
}

module.exports = {
  buildFdg,
}; 