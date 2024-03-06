export const indexRouteHandler: Deno.ServeHandler = function handler(
  _req,
  _info,
) {
  const body = `<!DOCTYPE html>
<html lang="en-GB">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Learn WGPU</title>
    <style>
      canvas {
        background-color: black;
      }
    </style>
  </head>
  <body>
    <main id="wasm-example">
      <script type="module">
        import init from "/learn_wgpu.js";
        init().then(() => {
          console.log("WASM Loaded");
        });
      </script>
    </main>
  </body>
</html>
  `;

  return new Response(body, {
    headers: { "content-type": "text/html; charset=utf-8" },
  });
};
