const traverse = require('@babel/traverse').default;
const { createKgNodeId, createHash } = require('../utils/idUtils');
const { jsxTagAllowList } = require('../config');

/**
 * Extracts JSX element information from a component's AST.
 * @param {object} ast - The AST of the component file.
 * @param {string} componentName - The name of the component.
 * @returns {object[]} An array of JSX element KG nodes.
 */
function extractJsxElements({ ast, componentName }) {
  const jsxElements = [];

  traverse(ast, {
    JSXOpeningElement(path) {
      const tagName = path.node.name.name;
      if (jsxTagAllowList.includes(tagName)) {
        const attributes = {};
        let elementId = null;

        path.get('attributes').forEach((attributePath) => {
          const attribute = attributePath.node;
          const attrName = attribute.name.name;
          const attrValue = attribute.value;

          if (attrValue) {
            if (attrValue.type === 'StringLiteral') {
              attributes[attrName] = attrValue.value;
            } else if (attrValue.type === 'JSXExpressionContainer') {
              attributes[attrName] = attributePath
                .get('value.expression')
                .toString();
            }
          }

          if (attrName === 'id') {
            elementId = attributes[attrName];
          }
        });

        if (!elementId) {
          const pos = `${path.node.start}-${path.node.end}`;
          elementId = createHash(`${componentName}:${tagName}:${pos}`);
        }

        const nodeId = createKgNodeId('JSXElement', `${tagName}:${elementId}`, componentName);
        jsxElements.push({
          id: nodeId,
          type: 'JSXElement',
          tagName,
          attributes,
          component: componentName,
        });
      }
    },
  });

  return jsxElements;
}

module.exports = {
  extractJsxElements,
}; 