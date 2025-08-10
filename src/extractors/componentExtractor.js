const traverse = require('@babel/traverse').default;
const path = require('path');
const { isComponent } = require('../utils/astUtils');
const { createKgNodeId } = require('../utils/idUtils');
/**
 * Attempts to find a root JSX element id (id attribute) for a component by name.
 * Returns the first encountered id on a returned JSX element or on an implicit
 * JSX body of an arrow function component.
 */
function findComponentDomId(ast, componentName) {
  let domId = null;
  traverse(ast, {
    FunctionDeclaration(p) {
      if (domId) return;
      if (p.node.id && p.node.id.name === componentName) {
        p.traverse({
          ReturnStatement(rp) {
            if (domId) return;
            const arg = rp.node.argument;
            if (arg && arg.type === 'JSXElement') {
              const opening = rp.get('argument.openingElement');
              const attrs = opening.get('attributes') || [];
              attrs.forEach((attrPath) => {
                const attr = attrPath.node;
                if (attr && attr.name && attr.name.name === 'id') {
                  const val = attr.value;
                  if (val && val.type === 'StringLiteral') domId = val.value;
                  else if (val && val.type === 'JSXExpressionContainer') domId = attrPath.get('value.expression').toString();
                }
              });
            }
          },
        });
      }
    },
    VariableDeclarator(p) {
      if (domId) return;
      if (p.node.id && p.node.id.name === componentName) {
        const init = p.node.init;
        if (init && init.type === 'ArrowFunctionExpression') {
          if (init.body && init.body.type === 'JSXElement') {
            const opening = p.get('init.body.openingElement');
            const attrs = opening.get('attributes') || [];
            attrs.forEach((attrPath) => {
              const attr = attrPath.node;
              if (attr && attr.name && attr.name.name === 'id') {
                const val = attr.value;
                if (val && val.type === 'StringLiteral') domId = val.value;
                else if (val && val.type === 'JSXExpressionContainer') domId = attrPath.get('value.expression').toString();
              }
            });
          } else if (init.body && init.body.type === 'BlockStatement') {
            // Look for return statements in block body
            p.get('init.body').traverse({
              ReturnStatement(rp) {
                if (domId) return;
                const arg = rp.node.argument;
                if (arg && arg.type === 'JSXElement') {
                  const opening = rp.get('argument.openingElement');
                  const attrs = opening.get('attributes') || [];
                  attrs.forEach((attrPath) => {
                    const attr = attrPath.node;
                    if (attr && attr.name && attr.name.name === 'id') {
                      const val = attr.value;
                      if (val && val.type === 'StringLiteral') domId = val.value;
                      else if (val && val.type === 'JSXExpressionContainer') domId = attrPath.get('value.expression').toString();
                    }
                  });
                }
              },
            });
          }
        }
      }
    },
  });
  return domId;
}

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
  const fileBaseName = path.basename(filePath);

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
            const domId = findComponentDomId(ast, componentName);
            components.push({
              id: createKgNodeId('Component', componentName, undefined, filePath),
              type: 'Component',
              name: componentName,
              filePath,
              exportType: 'named',
              props: [],
              hooksUsed: [],
              contextUsed: [],
              domId,
            });
          }
        } else if (declaration.type === 'VariableDeclaration') {
          declaration.declarations.forEach((declarator) => {
            if (isComponent(declarator.init)) {
              const componentName = declarator.id.name;
              if (!componentNames.has(componentName)) {
                componentNames.add(componentName);
                const domId = findComponentDomId(ast, componentName);
                components.push({
                  id: createKgNodeId('Component', componentName, undefined, filePath),
                  type: 'Component',
                  name: componentName,
                  filePath,
                  exportType: 'named',
                  props: [],
                  hooksUsed: [],
                  contextUsed: [],
                  domId,
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
          const domId = findComponentDomId(ast, componentName);
          components.push({
            id: createKgNodeId('Component', componentName, undefined, filePath),
            type: 'Component',
            name: componentName,
            filePath,
            exportType: 'default',
            props: [],
            hooksUsed: [],
            contextUsed: [],
            domId,
          });
        }
      } else if (declaration.type === 'Identifier') {
        // Find the declaration of the identifier
        const binding = path.scope.getBinding(declaration.name);
        if (binding && isComponent(binding.path.node)) {
            const componentName = declaration.name;
            if (!componentNames.has(componentName)) {
              componentNames.add(componentName);
              const domId = findComponentDomId(ast, componentName);
              components.push({
                  id: createKgNodeId('Component', componentName, undefined, filePath),
                  type: 'Component',
                  name: componentName,
                  filePath,
                  exportType: 'default',
                  props: [],
                  hooksUsed: [],
                  contextUsed: [],
                  domId,
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
              const domId = findComponentDomId(ast, componentName);
              components.push({
                id: createKgNodeId('Component', componentName, undefined, filePath),
                type: 'Component',
                name: componentName,
                filePath,
                exportType: 'default',
                props: [],
                hooksUsed: [],
                contextUsed: [],
                domId,
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