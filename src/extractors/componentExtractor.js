const traverse = require('@babel/traverse').default;
const { isComponent } = require('../utils/astUtils');
const { createKgNodeId } = require('../utils/idUtils');

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
          const componentName = declaration.id ? declaration.id.name : 'unnamed';
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
        const componentName = declaration.id ? declaration.id.name : 'default';
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