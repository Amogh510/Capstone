const traverse = require('@babel/traverse').default;
const { createKgNodeId } = require('../utils/idUtils');

/**
 * Extracts state information from an AST.
 * @param {object} ast - The AST to analyze.
 * @param {string} componentName - The name of the component being analyzed.
 * @returns {object[]} An array of state KG nodes.
 */
function extractStates({ ast, componentName }) {
  const states = [];

  traverse(ast, {
    CallExpression(path) {
      const { callee } = path.node;
      if (callee.name === 'useState' || callee.name === 'useReducer') {
        const declarator = path.findParent(
          (p) => p.isVariableDeclarator()
        );
        if (declarator) {
          if (declarator.node.id.type === 'ArrayPattern') {
            declarator.node.id.elements.forEach((element, index) => {
              if (index === 0) { // state variable
                const stateName = element.name;
                const initialValue =
                  path.node.arguments.length > 0
                    ? path.get('arguments.0').toString()
                    : null;
                states.push({
                  id: createKgNodeId('State', stateName, componentName),
                  type: 'State',
                  name: stateName,
                  declaredInComponent: componentName,
                  declarationType: callee.name,
                  initialValue,
                });
              }
            });
          } else {
            const stateName = declarator.node.id.name;
            const initialValue =
              path.node.arguments.length > 0
                ? path.get('arguments.0').toString()
                : null;
            states.push({
              id: createKgNodeId('State', stateName, componentName),
              type: 'State',
              name: stateName,
              declaredInComponent: componentName,
              declarationType: callee.name,
              initialValue,
            });
          }
        }
      }
    },
  });

  return states;
}

module.exports = {
  extractStates,
}; 