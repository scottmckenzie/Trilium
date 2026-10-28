import { createXai, type XaiProvider as XaiSDKProvider } from "@ai-sdk/xai";
import type { LlmAttachmentKind, LlmReasoningEffort } from "@triliumnext/commons";
import type { ToolSet } from "ai";

import type { LlmProviderConfig, ModelInfo } from "../types.js";
import { BaseProvider, type RemoteModel } from "./base_provider.js";
import { llmFetch } from "./fetch.js";

const OFFICIAL_BASE_URL = "https://api.x.ai/v1";

/**
 * Effort levels xAI accepts on models that take `reasoningEffort`.
 * {@link supportsXaiReasoningEffort} keeps the fixed `grok-4.20-*-reasoning` /
 * `*-non-reasoning` pair off the dropdown; those ids reject every value.
 */
const REASONING_EFFORTS = [
    "none", "low", "medium", "high", "xhigh"
] as const satisfies LlmReasoningEffort[];
type ReasoningEffort = (typeof REASONING_EFFORTS)[number];
/** xAI's default effort on models that reason (`grok-4.7`, `grok-4.3`, …). */
const DEFAULT_REASONING_EFFORT: ReasoningEffort = "high";

/** Non-chat ids from `/v1/models` (image, video, voice, embeddings). */
const NON_CHAT = /imagine|embedding|voice|tts|transcription|realtime|moderation/i;

/**
 * xAI Grok via `@ai-sdk/xai`. A named card so {@link BaseProvider.getProviderPrices}
 * can resolve bare `grok-*` ids against the committed price table.
 */
export class XaiProvider extends BaseProvider {
    name = "xai";
    protected defaultModel = "grok-4.7";
    /** Cheaper than the flagship; titles are one short line per new chat. */
    protected titleModel = "grok-4.3";

    private xai: XaiSDKProvider;

    constructor(apiKey: string, baseURL?: string) {
        super(apiKey, baseURL);
        if (!apiKey) {
            throw new Error("API key is required for xAI provider");
        }
        this.xai = createXai({
            apiKey,
            baseURL: this.baseURL ?? OFFICIAL_BASE_URL,
            fetch: llmFetch
        });
    }

    protected createModel(modelId: string) {
        return this.xai(modelId);
    }

    /** `/models` returns ids only, so {@link xaiModelName} builds the label. */
    protected override modelName(id: string): string {
        return xaiModelName(id);
    }

    /**
     * Lists chat models from `/models`. On the official endpoint, {@link isXaiChatModel}
     * drops image/video/voice/embedding ids; a custom `baseURL` lists everything.
     */
    protected override async fetchRemoteModels(): Promise<RemoteModel[] | null> {
        const payload = await this.fetchJson(`${this.baseURL ?? OFFICIAL_BASE_URL}/models`, {
            Authorization: `Bearer ${this.apiKey}`
        });
        const data = (payload as { data?: unknown }).data;
        if (!Array.isArray(data)) {
            throw new Error("Unexpected /models response shape");
        }
        const ids = data
            .filter((m): m is { id: string } => typeof (m as { id?: unknown }).id === "string")
            .map(m => m.id);
        const chatIds = this.baseURL ? ids : ids.filter(isXaiChatModel);
        return chatIds.map(id => ({ id, name: xaiModelName(id) }));
    }

    /** Attaches `reasoningEfforts` on models {@link supportsXaiReasoningEffort} allows. */
    override async listModels(): Promise<ModelInfo[]> {
        const models = await super.listModels();
        return models.map(m => supportsXaiReasoningEffort(m.id)
            ? {
                ...m,
                reasoningEfforts: [ ...REASONING_EFFORTS ],
                defaultReasoningEffort: DEFAULT_REASONING_EFFORT
            }
            : m);
    }

    /**
     * Sends `config.reasoningEffort`, or {@link DEFAULT_REASONING_EFFORT} when
     * unset, through `providerOptions.xai`. Skips models that reject the field.
     */
    protected override chatProviderOptions(config: LlmProviderConfig) {
        const modelId = config.model || this.defaultModel;
        if (!supportsXaiReasoningEffort(modelId)) {
            return undefined;
        }
        const effort = REASONING_EFFORTS.find(level => level === config.reasoningEffort)
            ?? DEFAULT_REASONING_EFFORT;
        return { xai: { reasoningEffort: effort } };
    }

    protected override addWebSearchTool(tools: ToolSet): void {
        // Same ToolSet cast as AnthropicProvider until `ai` and the provider
        // packages agree on the tool input type again.
        tools.web_search = this.xai.tools.webSearch() as ToolSet[string];
    }

    /** Chat models accept images; other kinds become `[attached …]` placeholders. */
    protected override acceptsAttachment(kind: LlmAttachmentKind, _modelId: string): boolean {
        return kind === "image";
    }

    /**
     * Pre-selects the newest numbered `grok-N` / `grok-N.M` id and `grok-latest`
     * when listed. Dated, build, code-fast and fixed reasoning pairs stay available
     * but unchecked.
     */
    override recommendedModelIds(models: ModelInfo[]): Set<string> {
        const numbered = models.filter(m => /^grok-\d+(?:\.\d+)?$/.test(m.id));
        const max = numbered.reduce((best, m) => Math.max(best, grokVersion(m.id)), 0);
        const newest = numbered.filter(m => grokVersion(m.id) === max).map(m => m.id);
        const latest = models.find(m => m.id === "grok-latest");
        return new Set(latest ? [ ...newest, latest.id ] : newest);
    }
}

/** Chat models only — image/video/voice/embedding ids stay out of the picker. */
export function isXaiChatModel(id: string): boolean {
    return !NON_CHAT.test(id);
}

/**
 * `grok-4.7` → "Grok 4.7", `grok-4.20-0309-reasoning` → "Grok 4.20 0309 Reasoning".
 * An unprefixed id (a proxy serving something else) is left as it came.
 */
export function xaiModelName(id: string): string {
    const match = /^grok-(.+)$/i.exec(id);
    if (!match) {
        return id;
    }
    const words = match[1].split("-").map(word => (
        /^\d+(?:\.\d+)?$/.test(word) ? word : word.charAt(0).toUpperCase() + word.slice(1)
    ));
    return `Grok ${words.join(" ")}`;
}

/**
 * Whether the model accepts `reasoningEffort`. Mirrors `@ai-sdk/xai`'s
 * `supportsReasoningEffort`: the fixed reasoning/non-reasoning 4.20 pair rejects
 * every value, including `none`.
 */
export function supportsXaiReasoningEffort(modelId: string): boolean {
    return !/^grok-4\.20(-\d{4})?-(non-)?reasoning$/.test(modelId);
}

/** `grok-4.7` → 4.7, `grok-3` → 3. Zero for an id with no version number. */
function grokVersion(id: string): number {
    const match = /^grok-(\d+(?:\.\d+)?)$/.exec(id);
    return match ? parseFloat(match[1]) : 0;
}
