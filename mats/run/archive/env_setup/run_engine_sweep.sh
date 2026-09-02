cd /home/wbc
for v in baseline lang_only prefix_caching no_prefix_caching kv_fp8 chunked_prefill spec_mtp; do
  echo "=========== BEGIN $v ==========="
  timeout 900 python /home/wbc/try_engine.py $v 2>&1
  echo "=========== END $v exit=$? ==========="
done
echo "ALL_VARIANTS_DONE"
