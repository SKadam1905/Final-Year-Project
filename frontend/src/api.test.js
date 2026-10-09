import { afterEach, describe, expect, it, vi } from "vitest";
import { api, onUnauthorized, session, socketRefused, withToken } from "./api.js";

const SESSION = { token: "abc.def", expires_at: Date.now() / 1000 + 3600, user: { username: "officer", role: "officer" } };

afterEach(() => {
  session.clear();
  vi.unstubAllGlobals();
});

describe("api session handling", () => {
  it("sends the token as a Bearer header and on image/download links", async () => {
    session.set(SESSION);
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify([]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await api.zones();
    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe("Bearer abc.def");
    expect(withToken("http://x/a.png")).toBe("http://x/a.png?token=abc.def");
    expect(withToken("http://x/a.png?v=1")).toBe("http://x/a.png?v=1&token=abc.def");
  });

  it("clears the session and reports the reason on a 401", async () => {
    session.set(SESSION);
    const handler = vi.fn();
    onUnauthorized(handler);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Your session has expired: please log in again." }), { status: 401 })));
    await expect(api.zones()).rejects.toThrow("Your session has expired");
    expect(handler).toHaveBeenCalledWith("Your session has expired: please log in again.");
    expect(session.get()).toBeNull();
  });

  it("keeps the first reason: later requests without a token do not overwrite 'expired'", async () => {
    session.set(SESSION);
    const handler = vi.fn();
    onUnauthorized(handler);
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Your session has expired: please log in again." }), { status: 401 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Not logged in: please log in." }), { status: 401 })));
    await expect(api.zones()).rejects.toThrow("expired");
    await expect(api.config()).rejects.toThrow("Not logged in");
    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler).toHaveBeenCalledWith("Your session has expired: please log in again.");
  });

  it("forgets an expired session and treats a refused socket like a 401", () => {
    session.set({ ...SESSION, expires_at: Date.now() / 1000 - 1 });
    expect(session.get()).toBeNull();
    session.set(SESSION);
    const handler = vi.fn();
    onUnauthorized(handler);
    expect(socketRefused(new Error("xhr poll error"))).toBe(false); // backend down: just disconnected
    expect(socketRefused(new Error("Not logged in: please log in."))).toBe(true);
    expect(handler).toHaveBeenCalledWith("Not logged in: please log in.");
  });
});
