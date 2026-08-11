file(REMOVE_RECURSE
  "../../devel/lib/libBHand.pdb"
  "../../devel/lib/libBHand.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/BHand.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
