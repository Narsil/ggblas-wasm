// Import our outputted wasm ES6 module
// Which, export default's, an initialization function
import init from "./pkg/ggblas_wasm.js";

const runWasm = async () => {
  // Instantiate our wasm module
  const helloWorld = await init("./pkg/ggblas_wasm_bg.wasm");

  // Call the Add function export from wasm, save the result
  helloWorld.run();
  helloWorld.run_f16();
  // helloWorld.run_f16_mixed();

  // Set the result onto the body
};
runWasm();
