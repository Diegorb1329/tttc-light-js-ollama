import { SourceRow } from "tttc-common/schema";

// sync version of sha256 hash to make strong unique urls
// that do not leak
const { createHash } = require("crypto");
function sha256(str: string) {
  return createHash("sha256").update(str).digest("hex");
}

export function uniqueSlug(str: string): string {
  // Convert to lowercase
  const lowerCaseStr = str.toLowerCase();
  // Replace non-letter characters with a dash
  const dashReplacedStr = lowerCaseStr.replace(/[^a-z]+/g, "-");
  // Replace multiple consecutive dashes with a single dash
  const singleDashStr = dashReplacedStr.replace(/-+/g, "-");
  // Trim dashes from the start and end
  const trimmedStr = singleDashStr.replace(/^-+|-+$/g, "");
  // Postfix a timestamp to the slug to make it unique
  const final = trimmedStr + "-" + Date.now();
  // finally hash!
  const sha256Final = sha256(final);
  return sha256Final;
}

export function formatData(data: any): SourceRow[] {
  const ID_COLS = ["id", "Id", "ID", "comment-id", "i"];
  const COMMENT_COLS = ["comment", "Comment", "comment-body"];
  if (!data || !data.length) {
    throw Error("Invalid or empty data file");
  }
  const keys = new Set(Object.keys(data[0]));
  const id_column = ID_COLS.find((x) => keys.has(x));
  const comment_column = COMMENT_COLS.find((x) => keys.has(x));
  if (!comment_column) {
    throw Error(
      `The csv file must contain a comment column (valid column names: ${COMMENT_COLS.join(", ")})`,
    );
  }
  return data.map((row: any, i: number) => {
    const id = String({ ...row, i }[id_column!]);
    const comment = row[comment_column];
    const res: SourceRow = { id, comment };
    if (keys.has("video")) res.video = row.video;
    if (keys.has("interview")) res.interview = row.interview;
    if (keys.has("timestamp")) res.timestamp = row.timestamp;
    return res;
  });
}

/**
 * Determines the correct API key to send to PyServer based on environment configuration.
 * This ensures compatibility with PyServer's OpenAI/OpenRouter detection logic.
 * 
 * Logic:
 * - If OPENAI_API_BASE_URL is set to OpenRouter endpoint, use OPENROUTER_API_KEY or fallback to OPENAI_API_KEY
 * - Otherwise, use OPENAI_API_KEY (standard OpenAI)
 */
export function getApiKeyForPyServer(env: {
  OPENAI_API_KEY?: string;
  OPENROUTER_API_KEY?: string;
  OPENAI_API_BASE_URL?: string;
}): string {
  const isOpenRouter = env.OPENAI_API_BASE_URL?.includes('openrouter.ai');
  
  if (isOpenRouter) {
    // When using OpenRouter, prefer OPENROUTER_API_KEY, fallback to OPENAI_API_KEY
    const apiKey = env.OPENROUTER_API_KEY || env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OPENROUTER_API_KEY or OPENAI_API_KEY must be provided when using OpenRouter");
    }
    console.log(">>> [EXPRESS-SERVER] Using OpenRouter API key for PyServer");
    return apiKey;
  } else {
    // When using OpenAI directly
    if (!env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY must be provided when using OpenAI directly");
    }
    console.log(">>> [EXPRESS-SERVER] Using OpenAI API key for PyServer");
    return env.OPENAI_API_KEY;
  }
}
