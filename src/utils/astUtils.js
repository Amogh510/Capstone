/**
 * Checks if a given AST node is a React component.
 * @param {object} node - The AST node to check.
 * @returns {boolean} - True if the node is a React component, false otherwise.
 */
function isComponent(node) {
  if (!node) return false;

  // Handle ArrowFunctionExpression with implicit return of JSXElement
  if (node.type === 'ArrowFunctionExpression' && node.body.type === 'JSXElement') {
    return true;
  }

  // Function declaration or arrow function that returns JSX
  if (
    (node.type === 'FunctionDeclaration' ||
      node.type === 'ArrowFunctionExpression') &&
    node.body &&
    node.body.type === 'BlockStatement'
  ) {
    let hasJSX = false;
    node.body.body.forEach((statement) => {
      if (
        statement.type === 'ReturnStatement' &&
        statement.argument &&
        statement.argument.type === 'JSXElement'
      ) {
        hasJSX = true;
      }
    });
    return hasJSX;
  }

  // Class component
  if (
    node.type === 'ClassDeclaration' &&
    node.superClass &&
    node.superClass.property &&
    (node.superClass.property.name === 'Component' ||
      node.superClass.property.name === 'PureComponent')
  ) {
    return true;
  }

  // Handle variable declarations with arrow functions
  if (node.type === 'VariableDeclaration') {
    return node.declarations.some(declarator => {
      if (declarator.init && isComponent(declarator.init)) {
        return true;
      }
      return false;
    });
  }

  // Handle variable declarator with arrow function
  if (node.type === 'VariableDeclarator' && node.init) {
    return isComponent(node.init);
  }

  return false;
}

module.exports = {
  isComponent,
}; 