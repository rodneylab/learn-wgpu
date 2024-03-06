import { serveDir } from "$std/http/file_server.ts";
import { indexRouteHandler } from "@/routes/index.ts";

const handler: Deno.ServeHandler = async function handler(req, info) {
  const { url } = req;
  const { pathname } = new URL(url);

  if (pathname === "/index.html" || pathname === "/") {
    return indexRouteHandler(req, info);
  }

  const response = await serveDir(req, { fsRoot: "public" });
  return response;
};

Deno.serve(handler);
