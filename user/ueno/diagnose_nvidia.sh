#!/usr/bin/env bash
set -euo pipefail

header() {
    echo
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "$1"
    echo "───────────────────────────────────────────────────────────────────────────────"
}

header "1) ホスト上で nvidia-smi を実行（ドライバ動作チェック）"
if command -v nvidia-smi &>/dev/null; then
    echo "→ nvidia-smi コマンドが見つかりました。実行結果："
    nvidia-smi || echo "⚠️  nvidia-smi の実行に失敗しました（終了コード: $?）"
else
    echo "⚠️  nvidia-smi コマンドが見つかりません。NVIDIA ドライバが入っていない可能性があります。"
fi

header "2) libnvidia-ml.so 系ファイルの存在場所とパーミッション確認"
LIB_PATHS=(
    "/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.247.01"
    "/usr/lib/nvidia-*/libnvidia-ml.so.*"
)
for path in "${LIB_PATHS[@]}"; do
    for f in $path; do
        if [ -e "$f" ]; then
            echo "🔍 $f :"
            ls -l "$f"
        fi
    done
done

header "3) ldconfig キャッシュに登録された libnvidia-ml.so.1 の一覧"
ldconfig -p | grep libnvidia-ml.so.1 || echo "→ ldconfig キャッシュに libnvidia-ml.so.1 が見つかりません。"

header "4) /etc/ld.so.conf.d 以下に NVIDIA 関連設定があるか確認"
if [ -d /etc/ld.so.conf.d ]; then
    grep -H "nvidia" /etc/ld.so.conf.d/*.conf 2>/dev/null || echo "→ NVIDIA 関連の .conf は見つかりませんでした。"
else
    echo "→ /etc/ld.so.conf.d ディレクトリが存在しません。"
fi

header "5) Docker が認識する Runtimes と Default Runtime を表示"
echo "→ docker info | grep -i 'Runtimes'"
docker info | grep -i "Runtimes" || echo "⚠️  Docker が Runtimes を取得できませんでした。"
echo "→ docker info | grep -i 'Default Runtime'"
docker info | grep -i "Default Runtime" || echo "⚠️  Docker が Default Runtime を取得できませんでした。"

header "6) /etc/docker/daemon.json の内容"
if [ -f /etc/docker/daemon.json ]; then
    echo "→ /etc/docker/daemon.json の内容："
    sed -n '1,200p' /etc/docker/daemon.json || true
else
    echo "→ /etc/docker/daemon.json は存在しません。"
fi

header "7) テスト用 GPU コンテナで nvidia-smi を実行"
TEST_IMAGE="nvidia/cuda:12.2.0-runtime-ubuntu22.04"
echo "→ イメージをプルしてみます： docker pull $TEST_IMAGE"
docker pull "$TEST_IMAGE" || echo "⚠️  イメージのプルに失敗しました（続行します）"
echo "→ docker run --rm --runtime=nvidia --gpus all $TEST_IMAGE nvidia-smi"
docker run --rm --runtime=nvidia --gpus all "$TEST_IMAGE" nvidia-smi || echo "⚠️  テストコンテナ内で nvidia-smi が失敗しました（エラー続行）"

header "8) 診断まとめ"
echo "1) ホスト上の nvidia-smi 実行結果を確認してください。"
echo "   - ここで GPU 情報が表示されていれば、ドライバは正常に動作しています。"
echo "2) libnvidia-ml.so 系ファイルのパーミッションを確認しました。"
echo "   - 実体ファイルに実行ビット（rwx）が付いているか要チェックです。"
echo "   - もし -rw-r--r-- (0644) のままなら、"
echo "       sudo chmod 755 <実ファイルパス> で修正してください。"
echo "3) ldconfig キャッシュに libnvidia-ml.so.1 が登録されているかを確認しました。"
echo "   - ldconfig -p | grep libnvidia-ml.so.1 で出力がない場合、"
echo "     /etc/ld.so.conf.d/ にパスを追加して sudo ldconfig するか、"
echo "     /usr/lib/x86_64-linux-gnu へシンボリックリンクを張る必要があります。"
echo "4) Docker のランタイム設定(nvidia) が認識されているかを確認しました。"
echo "   - Runtimes: に 'nvidia' が含まれていれば OK。"
echo "   - Default Runtime が runc でも構いませんが、compose で runtime: nvidia を指定する必要があります。"
echo "5) /etc/docker/daemon.json の内容を表示しました。"
echo "   - default-runtime: nvidia の設定があるか、runtimes: {nvidia: ...} の定義があるか確認してください。"
echo "6) テスト用コンテナ内で nvidia-smi を実行しました。"
echo "   - ここで GPU 情報が表示されれば、ホスト→コンテナ間のライブラリマウントは成功しています。"
echo "   - まだ 'libnvidia-ml.so.1 が見つからない' エラーが出る場合、手順2～3 を再確認ください。"
echo
echo "============================================================"
echo " NVIDIA Container Toolkit の診断が完了しました。"
echo "============================================================"
echo

exit 0
