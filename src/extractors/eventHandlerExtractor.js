const traverse = require('@babel/traverse').default;
const { createKgNodeId } = require('../utils/idUtils');
const path = require('path');

/**
 * Extracts event handler information from a component's AST.
 * @param {object} ast - The AST of the component file.
 * @param {string} componentName - The name of the component.
 * @returns {object[]} An array of event handler KG nodes.
 */
function extractEventHandlers({ ast, componentName, filePath }) {
  const eventHandlers = [];
  const fileBaseName = filePath ? path.basename(filePath) : undefined;
  const componentId = filePath ? createKgNodeId('Component', componentName, undefined, filePath) : undefined;

  traverse(ast, {
    JSXAttribute(path) {
      const attributeName = path.node.name.name;
      if (attributeName && attributeName.startsWith('on')) {
        const eventType = attributeName;
        const value = path.get('value');

        if (value.isJSXExpressionContainer()) {
          const expression = value.get('expression');
          if (expression.isIdentifier()) {
            const fnName = expression.node.name;
            const jsxElement = path.findParent((p) => p.isJSXOpeningElement());
            const attachedToJSX = jsxElement ? jsxElement.node.name.name : null;

            eventHandlers.push({
              id: createKgNodeId('EventHandler', fnName, componentName, filePath),
              type: 'EventHandler',
              name: fnName,
              eventType,
              definedInComponent: componentId || componentName,
              definedInComponentId: componentId,
              attachedToJSX,
            });
          }
        }
      }
    },
  });

  return eventHandlers;
}

module.exports = {
  extractEventHandlers,
}; 