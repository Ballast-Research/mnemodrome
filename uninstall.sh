#!/usr/bin/env bash
# Mnemodrome uninstaller — tears down the stack and lets you decide what happens
# to your captured memories.
#
# Usage:
#   ./uninstall.sh              — interactive (asks before removing anything)
#   ./uninstall.sh --keep-data  — remove containers + image, PRESERVE data (no prompts)
#   ./uninstall.sh --archive    — remove containers + image, ARCHIVE data to a tarball
#   ./uninstall.sh --purge      — remove containers + image AND delete data (destructive!)
#
# Your captured memories live under DATA_ROOT (all six layers: Neo4j graph, Qdrant
# vectors, Redis cache, event log, and the Markdown memory files). Uninstalling
# NEVER deletes them unless you explicitly choose to.
set -euo pipefail

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { printf "${CYAN}==> ${NC}%s\n" "$*"; }
success() { printf "${GREEN}\xe2\x9c\x93 ${NC}%s\n" "$*"; }
warn()    { printf "${YELLOW}\xe2\x9a\xa0  ${NC}%s\n" "$*"; }
error()   { printf "${RED}\xe2\x9c\x97  ${NC}%s\n" "$*" >&2; }
die()     { error "$*"; exit 1; }

MODE="interactive"
while [ $# -gt 0 ]; do
    case "$1" in
        --keep-data) MODE="keep" ;;
        --archive)   MODE="archive" ;;
        --purge)     MODE="purge" ;;
        -h|--help)   sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown arg: $1 (see --help)" ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[ -f docker-compose.yml ] || die "no docker-compose.yml here — run this from your Mnemodrome install directory."

# Locate the data. DATA_ROOT is written into .env by install.sh; fall back to ./data.
DATA_ROOT="$(grep -E '^DATA_ROOT=' .env 2>/dev/null | head -1 | cut -d= -f2- || true)"
[ -n "$DATA_ROOT" ] || DATA_ROOT="$SCRIPT_DIR/data"

echo ""
printf "${BOLD}Mnemodrome Uninstaller${NC}\n"
echo "────────────────────────────────────────────────────────"
info "Install dir : $SCRIPT_DIR"
info "Data root   : $DATA_ROOT"
if [ -d "$DATA_ROOT" ]; then
    SZ="$(du -sh "$DATA_ROOT" 2>/dev/null | cut -f1 || echo '?')"
    info "Data on disk: ${SZ} (your captured memories)"
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Stop and remove the containers (never touches data — it's bind-mounted)
# ---------------------------------------------------------------------------
info "Stopping and removing the Mnemodrome stack..."
docker compose --profile postgres down --remove-orphans 2>/dev/null || docker compose down --remove-orphans || true
success "Containers and network removed."

# ---------------------------------------------------------------------------
# 2. Remove the built server image (optional, safe — rebuilds on reinstall)
# ---------------------------------------------------------------------------
remove_image() {
    local img
    img="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -E '(^|/)mnemodrome-mcp-server:' | head -1 || true)"
    [ -z "$img" ] && img="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -iE 'mnemodrome.*mcp' | head -1 || true)"
    if [ -n "$img" ]; then docker rmi "$img" >/dev/null 2>&1 && success "Removed image $img" || warn "Could not remove image $img (in use?)"; else info "No built mnemodrome image found."; fi
}

# ---------------------------------------------------------------------------
# 3. Decide the fate of the captured memories
# ---------------------------------------------------------------------------
archive_data() {
    [ -d "$DATA_ROOT" ] || { warn "No data dir to archive."; return 0; }
    local stamp out
    stamp="$(date -u +%Y%m%d-%H%M%SZ 2>/dev/null || echo backup)"
    out="$SCRIPT_DIR/mnemodrome-data-${stamp}.tar.gz"
    info "Archiving your memories to: $out"
    tar -czf "$out" -C "$(dirname "$DATA_ROOT")" "$(basename "$DATA_ROOT")" \
        && success "Archived ($(du -h "$out" 2>/dev/null | cut -f1)). Restore later by extracting it back to DATA_ROOT." \
        || die "Archive failed — leaving your data in place, nothing deleted."
}
delete_data() {
    [ -d "$DATA_ROOT" ] || { info "No data dir to delete."; return 0; }
    rm -rf "$DATA_ROOT" && success "Deleted captured data at $DATA_ROOT" || warn "Could not fully delete $DATA_ROOT"
}

case "$MODE" in
    keep)
        remove_image
        success "Data preserved in place at $DATA_ROOT."
        ;;
    archive)
        remove_image
        archive_data && delete_data
        ;;
    purge)
        remove_image
        warn "--purge: permanently deleting captured data."
        delete_data
        ;;
    interactive)
        # Image
        printf "Remove the built Mnemodrome server image? [Y/n] "
        read -r ans; case "${ans:-Y}" in [Nn]*) info "Keeping image.";; *) remove_image;; esac
        echo ""
        # Data — the important one. Default is the SAFE choice (keep).
        printf "${BOLD}Your captured memories at %s${NC}\n" "$DATA_ROOT"
        echo "  What should happen to them?"
        echo "    [K] Keep in place   (default — nothing is deleted)"
        echo "    [A] Archive to a .tar.gz here, then remove the live data"
        echo "    [D] Delete permanently"
        printf "  Choose [K/a/d]: "
        read -r choice
        case "${choice:-K}" in
            [Aa]*) archive_data && delete_data ;;
            [Dd]*)
                printf "  ${RED}Type the data path to confirm permanent deletion${NC}\n  (%s): " "$DATA_ROOT"
                read -r confirm
                if [ "$confirm" = "$DATA_ROOT" ]; then delete_data; else warn "Path did not match — data left untouched."; fi
                ;;
            *) success "Keeping your memories in place at $DATA_ROOT." ;;
        esac
        ;;
esac

echo ""
echo "────────────────────────────────────────────────────────"
success "Mnemodrome uninstalled."
[ -f "$SCRIPT_DIR/.env" ] && warn "Note: .env (with generated passwords) is still here — remove it manually if you want it gone."
echo ""
