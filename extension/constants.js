// PhishNet shared constants (single source of truth).
//
// Loaded by the MV3 service worker via importScripts() and by the options
// page via a <script> tag, so the default backend cannot drift between them.

const PHISHNET_DEFAULT_BACKEND = "http://localhost:8000";
