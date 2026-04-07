export type ActivityType = 'navigation' | 'api' | 'search' | 'action';
export type ActivityStatus = 'pending' | 'success' | 'error' | 'info';

export interface ActivityEntry {
  id: string;
  ts: string;
  type: ActivityType;
  message: string;
  status?: ActivityStatus;
}

const STORAGE_KEY = 'dashboard_activity_log';
const STORAGE_VERSION_KEY = 'dashboard_activity_log_version';
const STORAGE_VERSION = '2';
const MAX_ENTRIES = 200;
const UPDATE_EVENT = 'dashboard-activity-log-updated';

function safeParse(raw: string | null): ActivityEntry[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as ActivityEntry[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function isLegacyNoisyEntry(entry: ActivityEntry): boolean {
  return (
    entry.type === 'api' ||
    entry.type === 'navigation' ||
    entry.type === 'search' ||
    /^(GET|POST|PUT|PATCH|DELETE)\s+/i.test(entry.message) ||
    /^Opened\s+\//i.test(entry.message)
  );
}

function ensureVersionMigration(): void {
  const version = localStorage.getItem(STORAGE_VERSION_KEY);
  if (version === STORAGE_VERSION) {
    return;
  }
  const migrated = safeParse(localStorage.getItem(STORAGE_KEY)).filter((entry) => !isLegacyNoisyEntry(entry));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(migrated));
  localStorage.setItem(STORAGE_VERSION_KEY, STORAGE_VERSION);
}

export function getActivityLog(): ActivityEntry[] {
  ensureVersionMigration();
  return safeParse(localStorage.getItem(STORAGE_KEY)).filter((entry) => !isLegacyNoisyEntry(entry));
}

export function addActivityLog(type: ActivityType, message: string, status?: ActivityStatus): string {
  const entry: ActivityEntry = {
    id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    ts: new Date().toISOString(),
    type,
    message,
    status,
  };
  const current = getActivityLog();
  const next = [entry, ...current].slice(0, MAX_ENTRIES);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  window.dispatchEvent(new CustomEvent(UPDATE_EVENT));
  return entry.id;
}

export function updateActivityLog(id: string, patch: Partial<Pick<ActivityEntry, 'message' | 'status'>>): void {
  const current = getActivityLog();
  const next = current.map((entry) => {
    if (entry.id !== id) return entry;
    return {
      ...entry,
      ...patch,
      ts: new Date().toISOString(),
    };
  });
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  window.dispatchEvent(new CustomEvent(UPDATE_EVENT));
}

export function clearActivityLog(): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify([]));
  window.dispatchEvent(new CustomEvent(UPDATE_EVENT));
}

export function onActivityLogUpdated(listener: () => void): () => void {
  const wrapped = () => listener();
  window.addEventListener(UPDATE_EVENT, wrapped);
  return () => window.removeEventListener(UPDATE_EVENT, wrapped);
}
