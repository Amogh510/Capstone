const traverse = require('@babel/traverse').default;
const { createKgNodeId, createHash } = require('../utils/idUtils');
const { jsxTagAllowList } = require('../config');
const path = require('path');

/**
 * Extracts JSX element information from a component's AST.
 * @param {object} ast - The AST of the component file.
 * @param {string} componentName - The name of the component.
 * @returns {object[]} An array of JSX element KG nodes.
 */
function extractJsxElements({ ast, componentName, filePath }) {
  const jsxElements = [];
  const fileBaseName = filePath ? path.basename(filePath) : undefined;
  const componentId = filePath ? createKgNodeId('Component', componentName, undefined, filePath) : undefined;

  traverse(ast, {
    JSXOpeningElement(path) {
      if (!path.node.name || !path.node.name.name) {
        return; // Skip elements without valid names
      }
      const tagName = path.node.name.name;
      {
        const attributes = {};
        const classNames = [];
        let inlineStyleExpr = null;
        let elementId = null;

        const attributesPath = path.get('attributes');
        if (!attributesPath || !Array.isArray(attributesPath)) {
          return;
        }
        attributesPath.forEach((attributePath) => {
          const attribute = attributePath.node;
          if (!attribute || !attribute.name || !attribute.name.name) {
            return; // Skip attributes without valid names
          }
          const attrName = attribute.name.name;
          const attrValue = attribute.value;

          if (attrValue) {
            if (attrValue.type === 'StringLiteral') {
              attributes[attrName] = attrValue.value;
              if (attrName === 'className') {
                attrValue.value.split(/\s+/).forEach((t) => t && classNames.push(t));
              }
            } else if (attrValue.type === 'JSXExpressionContainer') {
              attributes[attrName] = attributePath
                .get('value.expression')
                .toString();
              if (attrName === 'className') {
                try {
                  const text = attributePath.get('value.expression').toString();
                  text.replace(/[`'"+{}()]/g, ' ').split(/\s+/).forEach((t) => t && classNames.push(t));
                } catch (e) {}
              } else if (attrName === 'style') {
                try {
                  inlineStyleExpr = attributePath.get('value.expression').toString();
                } catch (e) {}
              }
            }
          }

          if (attrName === 'id') {
            elementId = attributes[attrName];
          }
        });

        // If JSX has id attribute, elementId will be that; otherwise fallback to positional hash
        if (!elementId) {
          const pos = `${path.node.start}-${path.node.end}`;
          elementId = createHash(`${componentName}:${tagName}:${pos}`);
        }

        const nodeId = createKgNodeId('JSXElement', `${tagName}:${elementId}`, componentName, filePath);
        jsxElements.push({
          id: nodeId,
          type: 'JSXElement',
          tagName,
          attributes,
          component: componentId || componentName,
          componentId,
          domElementId: elementId,
          classNames,
          inlineStyleExpr,
        });
      }
    },
  });

  return jsxElements;
}

module.exports = {
  extractJsxElements,
}; 