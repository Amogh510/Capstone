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
  
  // Create a set of valid file paths for quick lookup
  const validFilePaths = new Set(files.map(file => normalizePath(file)));
  
  console.log('Valid file paths:', Array.from(validFilePaths).slice(0, 5)); // Show first 5 for debugging

  files.forEach((filePath) => {
    const fileId = normalizePath(filePath);
    const fileType = path.extname(filePath).substring(1);
    const isStyle = ['css', 'scss', 'sass'].includes(fileType);
    const linesOfCode = fs.readFileSync(filePath, 'utf-8').split('\n').length;
    const imports = [];
    const exports = [];
    const stylesImported = [];

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
            
            console.log(`Import in ${fileId}: ${importPath} -> ${resolvedPath} (valid: ${validFilePaths.has(resolvedPath)})`);
            
            // Only include imports that reference actual files in the codebase
            if (validFilePaths.has(resolvedPath)) {
              if (/\.(css|scss|sass)$/.test(resolvedPath)) {
                stylesImported.push(resolvedPath);
              } else {
                imports.push(resolvedPath);
                edges.push({
                  from: fileId,
                  to: resolvedPath,
                  type: 'staticImport',
                });
                console.log(`  ✓ Added edge: ${fileId} -> ${resolvedPath}`);
              }
            } else {
              console.log(`  ✗ Skipped external import: ${resolvedPath}`);
            }
          },
          ExportNamedDeclaration(babelPath) {
            if (babelPath.node.declaration && babelPath.node.declaration.id) {
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
    
    // Use Sets to avoid duplicates in kgNodeRefs
    const kgNodeRefs = {};
    fileKgNodes.forEach((node) => {
      if (!kgNodeRefs[node.type]) {
        kgNodeRefs[node.type] = new Set();
      }
      kgNodeRefs[node.type].add(node.id);
    });
    
    // Convert Sets back to arrays
    Object.keys(kgNodeRefs).forEach(type => {
      kgNodeRefs[type] = Array.from(kgNodeRefs[type]);
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

  console.log(`FDG built: ${nodes.length} nodes, ${edges.length} edges`);
  return { nodes, edges };
}

module.exports = {
  buildFdg,
}; 