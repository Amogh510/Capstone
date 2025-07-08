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

  traverse(ast, {
    ExportNamedDeclaration(path) {
      const { declaration } = path.node;
      if (declaration) {
        if (isComponent(declaration)) {
          const componentName = declaration.id ? declaration.id.name : 'unnamed';
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
        } else if (declaration.type === 'VariableDeclaration') {
          declaration.declarations.forEach((declarator) => {
            if (isComponent(declarator.init)) {
              const componentName = declarator.id.name;
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
          });
        }
      }
    },
    ExportDefaultDeclaration(path) {
      const { declaration } = path.node;
      if (isComponent(declaration)) {
        const componentName = declaration.id ? declaration.id.name : 'default';
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
      } else if (declaration.type === 'Identifier') {
        // Find the declaration of the identifier
        const binding = path.scope.getBinding(declaration.name);
        if (binding && isComponent(binding.path.node)) {
            const componentName = declaration.name;
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
      } else if (declaration.type === 'VariableDeclaration') {
        // Handle default export of variable declaration
        declaration.declarations.forEach((declarator) => {
          if (isComponent(declarator.init)) {
            const componentName = declarator.id.name;
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
        });
      }
    },
    VariableDeclaration(path) {
      // Also check for components that might not be exported
      if (path.node.kind === 'const' || path.node.kind === 'let') {
        path.node.declarations.forEach((declarator) => {
          if (isComponent(declarator.init)) {
            const componentName = declarator.id.name;
            // Check if this component is already exported
            const isExported = path.parent.type === 'ExportDefaultDeclaration' || 
                              path.parent.type === 'ExportNamedDeclaration';
            if (!isExported) {
              components.push({
                id: createKgNodeId('Component', componentName),
                type: 'Component',
                name: componentName,
                filePath,
                exportType: 'none',
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