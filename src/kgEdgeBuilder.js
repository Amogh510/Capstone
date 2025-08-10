/**
 * Build intra-file KG edges between KG nodes derived from the AST.
 * For each file, we add edges between:
 * - Component -> Prop (componentHasProp)
 * - Component -> State (componentDeclaresState)
 * - Component -> Hook (componentUsesHook)
 * - Component -> EventHandler (componentDefinesEventHandler)
 * - Component -> JSXElement (componentRendersJsx)
 * - Route -> Component (routeRendersComponent) when declared in same file and names match
 *
 * Each edge includes { from, to, type, intraFile: true, fileId }.
 *
 * @param {Map<string, object[]>} kgNodesByFile - Map of fileId to KG nodes extracted from that file
 * @returns {object[]} edges
 */
function buildKgEdges(kgNodesByFile) {
  const edges = [];

  for (const [fileId, nodes] of kgNodesByFile.entries()) {
    const byId = new Map();
    const components = [];
    const props = [];
    const states = [];
    const hooks = [];
    const handlers = [];
    const jsxElems = [];
    const routes = [];

    nodes.forEach((n) => {
      if (!n || !n.id) return;
      byId.set(n.id, n);
      switch (n.type) {
        case 'Component': components.push(n); break;
        case 'Prop': props.push(n); break;
        case 'State': states.push(n); break;
        case 'Hook': hooks.push(n); break;
        case 'EventHandler': handlers.push(n); break;
        case 'JSXElement': jsxElems.push(n); break;
        case 'Route': routes.push(n); break;
        default: break;
      }
    });

    // Helper: add edge
    const add = (from, to, type) => {
      if (!from || !to) return;
      edges.push({ from, to, type, intraFile: true, fileId });
    };

    // Component -> Prop
    props.forEach((p) => {
      const compId = p.passedToComponent || p.passedToComponentId;
      add(compId, p.id, 'componentHasProp');
    });

    // Component -> State
    states.forEach((s) => {
      const compId = s.declaredInComponent || s.declaredInComponentId;
      add(compId, s.id, 'componentDeclaresState');
    });

    // Component -> Hook
    hooks.forEach((h) => {
      const compId = h.calledInComponent || h.calledInComponentId;
      add(compId, h.id, 'componentUsesHook');
    });

    // Component -> EventHandler
    handlers.forEach((h) => {
      const compId = h.definedInComponent || h.definedInComponentId;
      add(compId, h.id, 'componentDefinesEventHandler');
    });

    // Component -> JSXElement
    jsxElems.forEach((x) => {
      const compId = x.component || x.componentId;
      add(compId, x.id, 'componentRendersJsx');
    });

    // Route -> Component (within same file, name match)
    routes.forEach((r) => {
      if (!r.element) return;
      const match = components.find((c) => c.name === r.element || c.id === r.element);
      if (match) add(r.id, match.id, 'routeRendersComponent');
    });
  }

  return edges;
}

module.exports = { buildKgEdges };

