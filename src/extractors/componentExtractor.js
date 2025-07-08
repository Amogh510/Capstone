const traverse = require('@babel/traverse').default;
const path = require('path');
const { isComponent } = require('../utils/astUtils');
const { createKgNodeId } = require('../utils/idUtils');

/**
 * Generates a meaningful component name from file path when component is anonymous
 * @param {string} filePath - The file path
 * @returns {string} A meaningful component name
 */
function generateComponentNameFromPath(filePath) {
  const fileName = path.basename(filePath, path.extname(filePath));
  
  // If it's index.js, try to get the parent directory name
  if (fileName === 'index') {
    const dirName = path.basename(path.dirname(filePath));
    return dirName.charAt(0).toUpperCase() + dirName.slice(1);
  }
  
  // Convert kebab-case or snake_case to PascalCase
  return fileName
    .split(/[-_]/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join('');
}

/**
 * Extracts component information from an AST.
 * @param {object} ast - The AST to analyze.
 * @param {string} filePath - The path to the file being analyzed.
 * @returns {object[]} An array of component KG nodes.
 */
function extractComponents({ ast, filePath }) {
  const components = [];
  const componentNames = new Set(); // Track component names to avoid duplicates

  traverse(ast, {
    ExportNamedDeclaration(path) {
      const { declaration } = path.node;
      if (declaration) {
        if (isComponent(declaration)) {
          let componentName = 'unnamed';
          
          // Try to get a meaningful name
          if (declaration.id && declaration.id.name) {
            componentName = declaration.id.name;
          } else {
            // Try to derive name from file path
            componentName = generateComponentNameFromPath(filePath);
          }
          
          if (!componentNames.has(componentName)) {
            componentNames.add(componentName);
            components.push({
              id: createKgNodeId('Component', componentName),
              type: 'Component',
              name: componentName,
              filePath,
              exportType: 'named',
              props: [],
              hooksUsed: [],
              contextUsed: [],
            });
          }
        } else if (declaration.type === 'VariableDeclaration') {
          declaration.declarations.forEach((declarator) => {
            if (isComponent(declarator.init)) {
              const componentName = declarator.id.name;
              if (!componentNames.has(componentName)) {
                componentNames.add(componentName);
                components.push({
                  id: createKgNodeId('Component', componentName),
                  type: 'Component',
                  name: componentName,
                  filePath,
                  exportType: 'named',
                  props: [],
                  hooksUsed: [],
                  contextUsed: [],
                });
              }
            }
          });
        }
      }
    },
    ExportDefaultDeclaration(path) {
      const { declaration } = path.node;
      if (isComponent(declaration)) {
        let componentName = 'unnamed';
        
                  // Try to get a meaningful name
          if (declaration.id && declaration.id.name) {
            componentName = declaration.id.name;
          } else {
            // Try to derive name from file path
            componentName = generateComponentNameFromPath(filePath);
          }
        
        if (!componentNames.has(componentName)) {
          componentNames.add(componentName);
          components.push({
            id: createKgNodeId('Component', componentName),
            type: 'Component',
            name: componentName,
            filePath,
            exportType: 'default',
            props: [],
            hooksUsed: [],
            contextUsed: [],
          });
        }
      } else if (declaration.type === 'Identifier') {
        // Find the declaration of the identifier
        const binding = path.scope.getBinding(declaration.name);
        if (binding && isComponent(binding.path.node)) {
            const componentName = declaration.name;
            if (!componentNames.has(componentName)) {
              componentNames.add(componentName);
              components.push({
                  id: createKgNodeId('Component', componentName),
                  type: 'Component',
                  name: componentName,
                  filePath,
                  exportType: 'default',
                  props: [],
                  hooksUsed: [],
                  contextUsed: [],
              });
            }
        }
      } else if (declaration.type === 'VariableDeclaration') {
        // Handle default export of variable declaration
        declaration.declarations.forEach((declarator) => {
          if (isComponent(declarator.init)) {
            const componentName = declarator.id.name;
            if (!componentNames.has(componentName)) {
              componentNames.add(componentName);
              components.push({
                id: createKgNodeId('Component', componentName),
                type: 'Component',
                name: componentName,
                filePath,
                exportType: 'default',
                props: [],
                hooksUsed: [],
                contextUsed: [],
              });
            }
          }
        });
      }
    },
  });

  return components;
}

module.exports = {
  extractComponents,
}; 