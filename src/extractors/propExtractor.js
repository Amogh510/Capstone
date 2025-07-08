const traverse = require('@babel/traverse').default;
const { createKgNodeId } = require('../utils/idUtils');

/**
 * Extracts prop information from a component's AST.
 * @param {object} ast - The AST of the component file.
 * @param {string} componentName - The name of the component.
 * @param {string} filePath - The path of the file.
 * @returns {object[]} An array of prop KG nodes.
 */
function extractProps({ ast, componentName, filePath }) {
  const props = [];

  traverse(ast, {
    FunctionDeclaration(path) {
      if (path.node.id.name === componentName) {
        const componentNode = path.node;
        if (componentNode.params.length > 0) {
          const propsParam = componentNode.params[0];
          if (propsParam.type === 'ObjectPattern') {
            propsParam.properties.forEach((prop) => {
              props.push({
                id: createKgNodeId('Prop', prop.key.name, componentName),
                type: 'Prop',
                name: prop.key.name,
                passedToComponent: componentName,
                passedFromFile: filePath,
                valueType: 'unknown', // This is hard to determine statically
              });
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
        variableDeclarator.node.id.name === componentName
      ) {
        const componentNode = path.node;
        if (componentNode.params.length > 0) {
          const propsParam = componentNode.params[0];
          if (propsParam.type === 'ObjectPattern') {
            propsParam.properties.forEach((prop) => {
              props.push({
                id: createKgNodeId('Prop', prop.key.name, componentName),
                type: 'Prop',
                name: prop.key.name,
                passedToComponent: componentName,
                passedFromFile: filePath,
                valueType: 'unknown',
              });
            });
          }
        }
      }
    },
    MemberExpression(path) {
        if (path.node.object.name === 'props') {
            const propName = path.node.property.name;
            props.push({
                id: createKgNodeId('Prop', propName, componentName),
                type: 'Prop',
                name: propName,
                passedToComponent: componentName,
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