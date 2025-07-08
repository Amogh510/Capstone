const path = require('path');
const { walk } = require('./fileWalker');
const { parseFile } = require('./astParser');
const { outputDir } = require('./config');
const { writeJsonFile, normalizePath } = require('./utils/fileUtils');

const { extractComponents } = require('./extractors/componentExtractor');
const { extractStates } = require('./extractors/stateExtractor');
const { extractProps } = require('./extractors/propExtractor');
const { extractHooks } = require('./extractors/hookExtractor');
const { extractEventHandlers } = require('./extractors/eventHandlerExtractor');
const { extractJsxElements } = require('./extractors/jsxElementExtractor');
const { extractContexts } = require('./extractors/contextExtractor');
const { extractRoutes } = require('./extractors/routeExtractor');

const { buildFdg } = require('./fdgBuilder');
const { buildKg } = require('./kgNodeBuilder');

async function main() {
  console.log('Starting analysis...');
  const files = walk();
  const kgNodesByFile = new Map();

  for (const filePath of files) {
    const normalizedPath = normalizePath(filePath);
    const ast = parseFile(filePath);
    if (!ast) continue;

    const fileNodes = [];

    const components = extractComponents({ ast, filePath: normalizedPath });
    console.log(`File: ${normalizedPath} - Components found: ${components.length}`);
    if (components.length > 0) {
      components.forEach(comp => console.log(`  - ${comp.name} (${comp.exportType})`));
    }
    fileNodes.push(...components);

    components.forEach((component) => {
      const componentName = component.name;
      fileNodes.push(...extractStates({ ast, componentName }));
      fileNodes.push(...extractProps({ ast, componentName, filePath: normalizedPath }));
      fileNodes.push(...extractHooks({ ast, componentName }));
      fileNodes.push(...extractEventHandlers({ ast, componentName }));
      fileNodes.push(...extractJsxElements({ ast, componentName }));
    });

    fileNodes.push(...extractContexts({ ast, filePath: normalizedPath }));
    const { routes, declareRoutes } = extractRoutes({ ast, filePath: normalizedPath });
    fileNodes.push(...routes);

    kgNodesByFile.set(normalizedPath, fileNodes);
  }

  const fdg = buildFdg(files, kgNodesByFile);
  const kg = buildKg(kgNodesByFile);

  // Update declareRoutes in FDG
  fdg.nodes.forEach((node) => {
    const fileKgNodes = kgNodesByFile.get(node.fileId);
    if (fileKgNodes) {
      const hasRoutes = fileKgNodes.some((kgNode) => kgNode.type === 'Route');
      if (hasRoutes) {
        node.declareRoutes = true;
      }
    }
  });

  writeJsonFile(path.join(outputDir, 'fdg.json'), fdg);
  writeJsonFile(path.join(outputDir, 'kg.json'), kg);

  console.log('Analysis complete.');
}

main().catch((error) => console.error('Unhandled error:', error)); 