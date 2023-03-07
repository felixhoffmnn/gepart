library(feather)
# write_feather(data.frame(f$screen_name, f$user_id, f$status_id, f$created_at, f$text, f$is_retweet, f$lang, f$fullname, f$faction, f$name, f$social, f$economic, f$source, f$display_text_width, f$reply_to_status_id, f$reply_to_user_id, f$reply_to_screen_name, f$is_quote, f$favorite_count, f$retweet_count, f$quote_count, f$reply_count, f$ext_media_type, f$quoted_status_id), "~/Documents/data.feather")

# filtered_f <- data.frame(f, select = -c(f$hashtags, f$symbols, f$urls_url, f$urls_to.co, f$urls_expanded_url, f$media_url, f$media_t.co, f$media_expanded_url, f$media_type, f$ext_media_url, f$ext_media_t.co, f$ext_media_expanded_url, f$mentions_user_id, f$mentions_screen_name, f$geo_coords, f$coords_coords, f$bbox_coords))

# m3$hashtags <- apply(m3, 1, FUN = function(x) {
#   toString(x$hashtags)
# })

# m3$mentions_user_id <- apply(m3, 1, FUN = function(x) {
#   toString(x$mentions_user_id)
# })

# m3$mentions_screen_name <- apply(m3, 1, FUN = function(x) {
#   toString(x$mentions_screen_name)
# })

# m3$geo_coords <- apply(m3, 1, FUN = function(x) {
#   toString(x$geo_coords)
# })

# m3$coords_coords <- apply(m3, 1, FUN = function(x) {
#   toString(x$coords_coords)
# })

# m3$bbox_coords <- apply(m3, 1, FUN = function(x) {
#   toString(x$bbox_coords)
# })

m3 <- lapply(m3, function(x) {
  sapply(x, function(y) {
    toString(y)
  })
})

write_feather(data.frame(m3), "~/Documents/data.feather")
