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
            
            // Determine import type for better logging
            const specifiers = babelPath.node.specifiers || [];
            let importType = 'unknown';
            let importedItems = [];
            
            specifiers.forEach(specifier => {
              if (specifier.type === 'ImportDefaultSpecifier') {
                importType = 'default';
                importedItems.push(specifier.local.name);
              } else if (specifier.type === 'ImportSpecifier') {
                importType = 'named';
                const importedName = specifier.imported ? specifier.imported.name : specifier.local.name;
                importedItems.push(importedName);
              } else if (specifier.type === 'ImportNamespaceSpecifier') {
                importType = 'namespace';
                importedItems.push(specifier.local.name);
              }
            });
            
            console.log(`Import in ${fileId}: ${importPath} -> ${resolvedPath} (${importType}: ${importedItems.join(', ')}) (valid: ${validFilePaths.has(resolvedPath)})`);
            
            // Only include imports that reference actual files in the codebase
            if (validFilePaths.has(resolvedPath)) {
              if (/\.(css|scss|sass)$/.test(resolvedPath)) {
                stylesImported.push(resolvedPath);
                // Add edge for CSS imports
                edges.push({
                  from: fileId,
                  to: resolvedPath,
                  type: 'styleImport',
                });
                console.log(`  ✓ Added style edge: ${fileId} -> ${resolvedPath}`);
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
          CallExpression(babelPath) {
            // Handle dynamic imports: import('./module')
            if (babelPath.node.callee.type === 'Import') {
              const importArg = babelPath.node.arguments[0];
              if (importArg && importArg.type === 'StringLiteral') {
                const importPath = importArg.value;
                const resolvedPath = resolveAlias(
                  importPath,
                  path.dirname(filePath)
                );
                
                console.log(`Dynamic import in ${fileId}: ${importPath} -> ${resolvedPath} (valid: ${validFilePaths.has(resolvedPath)})`);
                
                if (validFilePaths.has(resolvedPath)) {
                  if (/\.(css|scss|sass)$/.test(resolvedPath)) {
                    stylesImported.push(resolvedPath);
                    edges.push({
                      from: fileId,
                      to: resolvedPath,
                      type: 'dynamicStyleImport',
                    });
                    console.log(`  ✓ Added dynamic style edge: ${fileId} -> ${resolvedPath}`);
                  } else {
                    imports.push(resolvedPath);
                    edges.push({
                      from: fileId,
                      to: resolvedPath,
                      type: 'dynamicImport',
                    });
                    console.log(`  ✓ Added dynamic edge: ${fileId} -> ${resolvedPath}`);
                  }
                } else {
                  console.log(`  ✗ Skipped external dynamic import: ${resolvedPath}`);
                }
              }
            }
          },
          ExportNamedDeclaration(babelPath) {
            const { declaration, specifiers } = babelPath.node;
            
            // Handle named exports with declarations
            if (declaration) {
              if (declaration.id) {
                exports.push(declaration.id.name);
              } else if (declaration.type === 'VariableDeclaration') {
                declaration.declarations.forEach(declarator => {
                  if (declarator.id && declarator.id.name) {
                    exports.push(declarator.id.name);
                  }
                });
              }
            }
            
            // Handle re-exports (export { x, y } from 'module')
            if (specifiers && specifiers.length > 0) {
              specifiers.forEach(specifier => {
                if (specifier.exported && specifier.exported.name) {
                  exports.push(specifier.exported.name);
                }
              });
            }
          },
          ExportDefaultDeclaration(babelPath) {
            const { declaration } = babelPath.node;
            
            if (declaration) {
              if (declaration.id && declaration.id.name) {
                exports.push(declaration.id.name);
              } else if (declaration.type === 'Identifier') {
                exports.push(declaration.name);
              } else if (declaration.type === 'VariableDeclaration') {
                declaration.declarations.forEach(declarator => {
                  if (declarator.id && declarator.id.name) {
                    exports.push(declarator.id.name);
                  }
                });
              } else {
                exports.push('default');
              }
            } else {
              exports.push('default');
            }
          },
          ExportAllDeclaration(babelPath) {
            // Handle export * from 'module'
            exports.push('*');
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