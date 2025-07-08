const traverse = require('@babel/traverse').default;
const { createKgNodeId } = require('../utils/idUtils');

/**
 * Extracts event handler information from a component's AST.
 * @param {object} ast - The AST of the component file.
 * @param {string} componentName - The name of the component.
 * @returns {object[]} An array of event handler KG nodes.
 */
function extractEventHandlers({ ast, componentName }) {
  const eventHandlers = [];

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
              id: createKgNodeId('EventHandler', fnName, componentName),
              type: 'EventHandler',
              name: fnName,
              eventType,
              definedInComponent: componentName,
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