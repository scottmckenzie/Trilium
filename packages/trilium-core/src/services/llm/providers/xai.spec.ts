import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { createXaiMock, modelMock, webSearchMock } = vi.hoisted(() => ({
    createXaiMock: vi.fn(),
    modelMock: vi.fn((modelId: string) => ({ modelId })),
    webSearchMock: vi.fn(() => ({ kind: "web_search" }))
}));

vi.mock("@ai-sdk/xai", () => ({
    createXai: (opts: unknown) => {
        createXaiMock(opts);
        const fn: any = (modelId: string) => modelMock(modelId);
        fn.tools = { webSearch: webSearchMock };
        return fn;
    }
}));

const { streamTextMock } = vi.hoisted(() => ({
    streamTextMock: vi.fn((..._args: any[]) => ({}) as any)
}));

vi.mock("ai", async (importOriginal) => {
    const actual = await importOriginal<typeof import("ai")>();
    return { ...actual, streamText: streamTextMock };
});

import { installGlobalFetchAsApiTransport } from "../../../test/request_provider.js";
import { llmFetch } from "./fetch.js";
import { isXaiChatModel, supportsXaiReasoningEffort, XaiProvider, xaiModelName } from "./xai.js";

beforeEach(installGlobalFetchAsApiTransport);

describe("XaiProvider construction", () => {
    beforeEach(() => {
        createXaiMock.mockClear();
    });

    it("points at the official endpoint unless overridden, and requires a key", () => {
        new XaiProvider("xai-key");
        expect(createXaiMock).toHaveBeenCalledWith({
            apiKey: "xai-key",
            baseURL: "https://api.x.ai/v1",
            fetch: llmFetch
        });

        new XaiProvider("xai-key", "https://gateway.example/v1");
        expect(createXaiMock).toHaveBeenLastCalledWith({
            apiKey: "xai-key",
            baseURL: "https://gateway.example/v1",
            fetch: llmFetch
        });

        new XaiProvider("xai-key", "");
        expect(createXaiMock).toHaveBeenLastCalledWith({
            apiKey: "xai-key",
            baseURL: "https://api.x.ai/v1",
            fetch: llmFetch
        });

        expect(() => new XaiProvider("")).toThrow(/API key is required/);
    });
});

describe("XaiProvider chat", () => {
    beforeEach(() => {
        streamTextMock.mockClear();
        webSearchMock.mockClear();
        modelMock.mockClear();
    });

    it("createModel uses the configured model id, falling back to the default", () => {
        const provider = new XaiProvider("xai-key");
        provider.chat([{ role: "user", content: "hi" }], {});
        expect(modelMock).toHaveBeenLastCalledWith("grok-4.7");

        provider.chat([{ role: "user", content: "hi" }], { model: "grok-4.3" });
        expect(modelMock).toHaveBeenLastCalledWith("grok-4.3");
    });

    it("adds the xAI web_search tool when web search is enabled", () => {
        const provider = new XaiProvider("xai-key");
        provider.chat([{ role: "user", content: "hi" }], { enableWebSearch: true });

        expect(webSearchMock).toHaveBeenCalledOnce();
        const opts = streamTextMock.mock.calls[0][0] as any;
        expect(opts.tools.web_search).toEqual({ kind: "web_search" });
        expect(opts.toolChoice).toBe("auto");
        expect(opts.stopWhen).toBeDefined();
    });

    it("sends reasoningEffort on models that accept it, and omits it on the fixed 4.20 pair", () => {
        const provider = new XaiProvider("xai-key");

        provider.chat([{ role: "user", content: "hi" }], { model: "grok-4.7", reasoningEffort: "low" });
        expect((streamTextMock.mock.calls.at(-1)?.[0] as any).providerOptions).toEqual({
            xai: { reasoningEffort: "low" }
        });

        provider.chat([{ role: "user", content: "hi" }], { model: "grok-4.20-reasoning" });
        expect((streamTextMock.mock.calls.at(-1)?.[0] as any).providerOptions).toBeUndefined();
    });
});

describe("XaiProvider model listing", () => {
    const fetchMock = vi.fn();
    const okJson = (body: unknown) => ({ ok: true, json: async () => body });

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal("fetch", fetchMock);
    });
    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it("lists /models with a bearer key, drops non-chat ids, and prices against the table", async () => {
        fetchMock.mockResolvedValue(okJson({
            data: [
                { id: "grok-4.7" },
                { id: "grok-4.3" },
                { id: "grok-imagine-image" },
                { id: "grok-4.20-reasoning" }
            ]
        }));
        const provider = new XaiProvider("xai-key");

        const models = await provider.listModels();
        expect(fetchMock).toHaveBeenCalledWith(
            "https://api.x.ai/v1/models",
            expect.objectContaining({ headers: { Authorization: "Bearer xai-key" } })
        );
        // Curated ids keep price-table order (alphabetical); the imagine id is dropped.
        expect(models.map(m => m.id)).toEqual([ "grok-4.20-reasoning", "grok-4.3", "grok-4.7" ]);
        expect(models.find(m => m.id === "grok-4.7")).toMatchObject({
            pricing: { input: 2, output: 6 },
            contextWindow: 500000,
            reasoningEfforts: [ "none", "low", "medium", "high", "xhigh" ],
            defaultReasoningEffort: "high",
            attachmentKinds: [ "image" ]
        });
        expect(models.find(m => m.id === "grok-4.20-reasoning")).not.toHaveProperty("reasoningEfforts");
    });

    it("keeps every id on a custom baseURL", async () => {
        fetchMock.mockResolvedValue(okJson({ data: [{ id: "grok-imagine-image" }, { id: "custom-model" }] }));
        const models = await new XaiProvider("xai-key", "https://gateway.example/v1").listModels();
        // Unknown remotes are sorted alphabetically after any curated hit.
        expect(models.map(m => m.id)).toEqual([ "custom-model", "grok-imagine-image" ]);
    });

    it("recommends the newest numbered Grok and grok-latest when listed", () => {
        const provider = new XaiProvider("xai-key");
        const ids = [ "grok-4.5", "grok-4.7", "grok-4.3", "grok-4.20-reasoning", "grok-latest", "grok-build-0.1" ]
            .map(id => ({ id, name: id }));
        expect([ ...provider.recommendedModelIds(ids) ].sort()).toEqual([ "grok-4.7", "grok-latest" ]);
    });
});

describe("xai helpers", () => {
    it("names models from their ids", () => {
        expect(xaiModelName("grok-4.7")).toBe("Grok 4.7");
        expect(xaiModelName("grok-4.20-0309-reasoning")).toBe("Grok 4.20 0309 Reasoning");
        expect(xaiModelName("other")).toBe("other");
    });

    it("filters non-chat ids and detects reasoning-effort support", () => {
        expect(isXaiChatModel("grok-4.7")).toBe(true);
        expect(isXaiChatModel("grok-imagine-video")).toBe(false);
        expect(supportsXaiReasoningEffort("grok-4.7")).toBe(true);
        expect(supportsXaiReasoningEffort("grok-4.20-reasoning")).toBe(false);
        expect(supportsXaiReasoningEffort("grok-4.20-0309-non-reasoning")).toBe(false);
    });
});
