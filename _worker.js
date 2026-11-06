/**
 * Colorize Old — Pages advanced-mode worker.
 * /checkout and /buy keep the existing Stripe Checkout session (do not mint another).
 * /models/colorization_deploy_v2.onnx streams the same mirror.opencv.ai object as production.
 * Extensionless evergreen URLs map to their .html files.
 */
const STRIPE_CHECKOUT = "https://checkout.stripe.com/c/pay/cs_live_a1T4SfYXYsEtup2RzQxAkkCBilnMaE5naqQvWklnP6ugMH0ZXi3FjE5Zty#fidnandhYHdWcXxpYCc%2FJ2FgY2RwaXEnKSdicyc%2FNSknZHVsTmB8Jz8ndW5aaWxzYFowNFFNMW5cQHxAXHdRSH1LbGFPcEdnMWJKcnNdaE89amhLSn1qV2l1MzdVbHdTVVxgVVN1U0xNdUBJckMzaFxKRjJDUW1cQDRnVXFSMlZNbnVWSUd%2FUFZAdzU1bX1KNmlhc2MnKSdjd2poVmB3c2B3Jz9xd3BgKSdnZGZuYndqcGthRmppancnPycmY2NjY2NjJyknaWR8anBxUXx1YCc%2FJ3Zsa2JpYFpscWBoJ3gl";

const MODEL_PATH = "/models/colorization_deploy_v2.onnx";
const MODEL_UPSTREAM = "https://mirror.opencv.ai/colorization_deploy_v2.onnx";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    let path = url.pathname;
    if (path.length > 1 && path.endsWith("/")) path = path.slice(0, -1);

    if (path === "/checkout" || path === "/buy") {
      return new Response(null, {
        status: 302,
        headers: { location: STRIPE_CHECKOUT },
      });
    }

    if (path === MODEL_PATH) return proxyModel(request);

    if (path.endsWith(".html") && path !== "/index.html") {
      const asset = await fetchAsset(env, path);
      if (asset) {
        if (asset.body) await asset.body.cancel();
        const dest = new URL(request.url);
        dest.pathname = path.slice(0, -5) || "/";
        return Response.redirect(dest.toString(), 301);
      }
    }

    const candidates = [path];
    if (path === "/") candidates.push("/index.html");
    else if (!path.split("/").pop().includes(".")) candidates.push(path + ".html");

    const httpMethod = request["method"];
    for (const cand of candidates) {
      const asset = await fetchAsset(env, cand, httpMethod);
      if (asset) return decorate(asset, path);
    }

    const last = path.split("/").pop() || "";
    const extensionless = !last.includes(".");
    if (extensionless || path.endsWith(".html")) {
      const home = await fetchAsset(env, "/index.html", httpMethod);
      if (home) return decorate(home, "/", { fallback: true });
    }

    return new Response("Not found", {
      status: 404,
      headers: {
        "content-type": "text/plain; charset=utf-8",
        "cache-control": "public, max-age=60",
        "x-content-type-options": "nosniff",
      },
    });
  },
};

async function fetchAsset(env, pathname, method) {
  const req = new Request("https://assets.local" + pathname, {
    method: method === "HEAD" ? "HEAD" : "GET",
  });
  const res = await env.ASSETS.fetch(req);
  if (res.status === 404) {
    if (res.body) await res.body.cancel();
    return null;
  }
  return res;
}

function decorate(res, requestPath, opts) {
  const headers = new Headers(res.headers);
  const ct = (headers.get("content-type") || "").toLowerCase();
  const fallback = opts && opts.fallback;
  let cache = "public, max-age=300";
  let nosniff = false;
  if (fallback || requestPath === "/" || requestPath === "/index.html") {
    cache = "public, max-age=60";
    nosniff = true;
  } else if (requestPath === "/sitemap.xml") {
    cache = "public, max-age=3600";
  } else if (requestPath === "/robots.txt" || ct.includes("javascript") || requestPath.endsWith(".js")) {
    cache = "public, max-age=14400";
    nosniff = ct.includes("javascript") || requestPath.endsWith(".js");
  } else if (ct.startsWith("image/")) {
    cache = "public, max-age=86400";
  }
  headers.set("cache-control", cache);
  if (nosniff) headers.set("x-content-type-options", "nosniff");
  else headers.delete("x-content-type-options");
  return new Response(res.body, { status: res.status, headers });
}

async function proxyModel(request) {
  const cors = {
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "range, content-type",
    "access-control-allow-methods": "GET, HEAD, OPTIONS",
    "access-control-max-age": "86400",
  };
  const httpMethod = request["method"];
  if (httpMethod === "OPTIONS") {
    return new Response(null, { status: 204, headers: cors });
  }
  if (httpMethod !== "GET" && httpMethod !== "HEAD") {
    return new Response("Method not allowed", { status: 405, headers: { ...cors, allow: "GET, HEAD, OPTIONS" } });
  }
  const headers = new Headers();
  const range = request.headers.get("range");
  if (range) headers.set("range", range);
  const upstream = await fetch(MODEL_UPSTREAM, { method: httpMethod, headers });
  const out = new Headers(cors);
  for (const name of ["content-type", "content-length", "content-range", "accept-ranges", "etag", "last-modified"]) {
    const value = upstream.headers.get(name);
    if (value) out.set(name, value);
  }
  out.set("cache-control", "public, max-age=86400, immutable");
  return new Response(httpMethod === "HEAD" ? null : upstream.body, {
    status: upstream.status,
    headers: out,
  });
}
