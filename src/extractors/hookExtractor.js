const traverse = require('@babel/traverse').default;
const { createKgNodeId } = require('../utils/idUtils');

/**
 * Extracts hook information from a component's AST.
 * @param {object} ast - The AST of the component file.
 * @param {string} componentName - The name of the component.
 * @returns {object[]} An array of hook KG nodes.
 */
function extractHooks({ ast, componentName }) {
  const hooks = [];
  const reactHooks = new Set(['useEffect', 'useContext', 'useMemo', 'useCallback', 'useRef']);

  traverse(ast, {
    CallExpression(path) {
      const { callee } = path.node;
      const hookName = callee.name;

      if (hookName && hookName.startsWith('use')) {
        const isCustom = !reactHooks.has(hookName) && hookName !== 'useState' && hookName !== 'useReducer';
        const params = path.get('arguments').map((arg) => arg.toString());

        hooks.push({
          id: createKgNodeId('Hook', hookName, componentName),
          type: 'Hook',
          hookName,
          isCustom,
          params,
          calledInComponent: componentName,
        });
      }
    },
  });

  return hooks;
}

module.exports = {
  extractHooks,
}; 