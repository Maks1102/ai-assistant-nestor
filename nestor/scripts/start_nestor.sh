#!/usr/bin/env bash
set -eo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_PATH="$(cd "$SCRIPT_PATH/.." && pwd)"

# Проверка наличия tmux
if ! command -v tmux &> /dev/null; then
  echo "Ошибка: tmux не установлен."
  echo "Установите: sudo apt install tmux (Ubuntu/Debian) или sudo pacman -S tmux (Arch)"
  exit 1
fi

NESTOR_PATH="$PROJECT_ROOT_PATH"/install/nestor
SC_MACHINE_PATH="$PROJECT_ROOT_PATH"/install/sc-machine
FIXED_SEARCH_STRATEGY_TEMPLATE_PROCESSING_LIB="$PROJECT_ROOT_PATH"/install/fixed-search-strategy-template-processing-lib

LD_LIBRARY_PATH="$LD_LIBRARY_PATH/lib:$SC_MACHINE_PATH/lib:$FIXED_SEARCH_STRATEGY_TEMPLATE_PROCESSING_LIB/lib:$LD_LIBRARY_PATH"

SESSION_NAMES=()

# ============================================================
# Функции управления
# ============================================================

cleanup() {
  echo "Остановка сервисов..."
  for session in "${SESSION_NAMES[@]}"; do
    tmux kill-session -t "$session" 2>/dev/null || true
  done
  echo "Все сервисы остановлены."
}

stop_all() {
  echo "=== Остановка всех tmux-сессий nestor ==="
  for session in sc-machine sc-web interface; do
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session" 2>/dev/null || true
      echo "  Сессия '$session' остановлена."
    else
      echo "  Сессия '$session' не найдена, пропуск."
    fi
  done
  echo "Все сервисы остановлены."
}

build_kb() {
  echo "=== Сборка базы знаний ==="
  "$PROJECT_ROOT_PATH"/install/sc-machine/bin/sc-builder -i repo-patch.path -o kb.bin --clear
  "$PROJECT_ROOT_PATH"/install/sc-machine/bin/sc-builder -i repo.path -o kb.bin
  echo "База знаний успешно собрана."
}

start_machine() {
  echo "=== Запуск sc-machine ==="
  tmux new-session -d -s sc-machine "bash -c \"cd '$PROJECT_ROOT_PATH' && LD_LIBRARY_PATH='$LD_LIBRARY_PATH' $SC_MACHINE_PATH/bin/sc-machine -s kb.bin -e '$SC_MACHINE_PATH/lib/extensions;$NESTOR_PATH/lib/extensions' -c nestor.ini; read -p 'Нажмите Enter чтобы закрыть...'\""
  SESSION_NAMES+=(sc-machine)
  echo "sc-machine запущен в tmux-сессии 'sc-machine'"
  sleep 2
}

start_web() {
  echo "=== Запуск sc-web ==="
  tmux new-session -d -s sc-web "bash -c \"cd '$PROJECT_ROOT_PATH/sc-web' && source .venv/bin/activate && python3 server/app.py; read -p 'Нажмите Enter чтобы закрыть...'\""
  SESSION_NAMES+=(sc-web)
  echo "sc-web запущен в tmux-сессии 'sc-web'"
  sleep 2
}

start_interface() {
  echo "=== Запуск интерфейса ==="
  tmux new-session -d -s interface "bash -c \"cd '$PROJECT_ROOT_PATH/interface' && npm run start; read -p 'Нажмите Enter чтобы закрыть...'\""
  SESSION_NAMES+=(interface)
  echo "Интерфейс запущен в tmux-сессии 'interface'"
}

start_all() {
  build_kb
  start_machine
  start_web
  start_interface

  echo ""
  echo "Все сервисы запущены в tmux-сессиях:"
  for session in "${SESSION_NAMES[@]}"; do
    echo "  tmux attach-session -t $session"
  done
  echo ""
  echo "Для просмотра логов: tmux attach-session -t <имя>"
  echo "Для остановки: Ctrl+C или tmux kill-session -t <имя>"
  echo ""
}

restart_kb() {
  echo "============================================="
  echo "  ПЕРЕЗАПУСК БАЗЫ ЗНАНИЙ (build_kb + machine)"
  echo "============================================="
  stop_all
  sleep 1
  build_kb
  start_machine
  echo ""
  echo "База знаний пересобрана, sc-machine перезапущен."
  echo "Для остановки: tmux kill-session -t sc-machine"
  echo ""
}

restart_all() {
  echo "============================================="
  echo "  ПОЛНАЯ ПЕРЕЗАГРУЗКА СИСТЕМЫ"
  echo "============================================="
  stop_all
  sleep 1
  SESSION_NAMES=()
  start_all
  echo "Полная перезагрузка завершена."
}

# ============================================================
# Точка входа
# ============================================================

usage() {
  echo "Использование: $0 [команда]"
  echo ""
  echo "Команды:"
  echo "  (нет аргументов)  Полный запуск системы (сборка KB + все сервисы)"
  echo "  restart_kb        Перезапуск базы знаний (build_kb + sc-machine)"
  echo "  restart_all       Полная перезагрузка системы (остановка + перезапуск)"
  echo "  stop              Остановка всех сервисов"
  echo ""
}

trap cleanup EXIT INT TERM

case "${1:-}" in
  restart_kb)
    restart_kb
    read -r -p "Нажмите Enter для остановки сервисов..."
    ;;
  restart_all)
    restart_all
    read -r -p "Нажмите Enter для остановки сервисов..."
    ;;
  stop)
    stop_all
    ;;
  "")
    start_all
    read -r -p "Нажмите Enter для остановки всех сервисов..."
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Неизвестная команда: $1"
    usage
    exit 1
    ;;
esac
