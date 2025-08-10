const traverse = require('@babel/traverse').default;
const { createKgNodeId } = require('../utils/idUtils');
const path = require('path');

/**
 * Extracts prop information from a component's AST.
 * @param {object} ast - The AST of the component file.
 * @param {string} componentName - The name of the component.
 * @param {string} filePath - The path of the file.
 * @returns {object[]} An array of prop KG nodes.
 */
function extractProps({ ast, componentName, filePath }) {
  const props = [];
  const fileBaseName = filePath ? path.basename(filePath) : undefined;
  const componentId = filePath ? createKgNodeId('Component', componentName, undefined, filePath) : undefined;

  traverse(ast, {
    FunctionDeclaration(path) {
      if (path.node.id && path.node.id.name === componentName) {
        const componentNode = path.node;
        if (componentNode.params.length > 0) {
          const propsParam = componentNode.params[0];
          if (propsParam.type === 'ObjectPattern' && Array.isArray(propsParam.properties)) {
            propsParam.properties.forEach((prop) => {
              if (prop && prop.key && prop.key.name) {
                props.push({
                  id: createKgNodeId('Prop', prop.key.name, componentName, filePath),
                  type: 'Prop',
                  name: prop.key.name,
                  passedToComponent: componentId || componentName,
                  passedToComponentName: componentName,
                  passedFromFile: filePath,
                  valueType: 'unknown', // This is hard to determine statically
                });
              }
            });
          }
        }
      }
    },
    ArrowFunctionExpression(path) {
      const variableDeclarator = path.findParent((p) =>
        p.isVariableDeclarator()
      );
      if (
        variableDeclarator &&
        variableDeclarator.node.id &&
        variableDeclarator.node.id.name === componentName
      ) {
        const componentNode = path.node;
        if (componentNode.params.length > 0) {
          const propsParam = componentNode.params[0];
          if (propsParam.type === 'ObjectPattern' && Array.isArray(propsParam.properties)) {
            propsParam.properties.forEach((prop) => {
              if (prop && prop.key && prop.key.name) {
                props.push({
                  id: createKgNodeId('Prop', prop.key.name, componentName, filePath),
                  type: 'Prop',
                  name: prop.key.name,
                  passedToComponent: componentId || componentName,
                  passedToComponentName: componentName,
                  passedFromFile: filePath,
                  valueType: 'unknown',
                });
              }
            });
          }
        }
      }
    },
    MemberExpression(path) {
        if (path.node.object && path.node.object.name === 'props' && path.node.property && path.node.property.name) {
            const propName = path.node.property.name;
            props.push({
                id: createKgNodeId('Prop', propName, componentName, filePath),
                type: 'Prop',
                name: propName,
                passedToComponent: componentId || componentName,
                passedToComponentName: componentName,
                passedFromFile: filePath,
                valueType: 'unknown',
            });
        }
    }
  });

  return props;
}

module.exports = {
  extractProps,
}; 