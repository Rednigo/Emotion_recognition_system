package com.google.mediapipe.examples.facelandmarker.feed

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.google.android.exoplayer2.ExoPlayer
import com.google.android.exoplayer2.MediaItem
import com.google.mediapipe.examples.facelandmarker.R
import com.google.mediapipe.examples.facelandmarker.databinding.ItemFeedPhotoBinding
import com.google.mediapipe.examples.facelandmarker.databinding.ItemFeedVideoBinding
import com.google.mediapipe.examples.facelandmarker.network.FeedItem

class FeedAdapter(
    private val onLikeClick: (FeedItem) -> Unit
) : ListAdapter<FeedItem, RecyclerView.ViewHolder>(FeedDiffCallback()) {

    companion object {
        private const val TYPE_PHOTO = 0
        private const val TYPE_VIDEO = 1
    }

    private val likedItems = mutableSetOf<String>()
    private val activeExoPlayers = mutableMapOf<String, ExoPlayer>()

    override fun getItemViewType(position: Int): Int {
        return when (getItem(position).type) {
            "photo" -> TYPE_PHOTO
            "video" -> TYPE_VIDEO
            else -> TYPE_PHOTO
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return when (viewType) {
            TYPE_PHOTO -> PhotoViewHolder(
                ItemFeedPhotoBinding.inflate(inflater, parent, false)
            )
            TYPE_VIDEO -> VideoViewHolder(
                ItemFeedVideoBinding.inflate(inflater, parent, false)
            )
            else -> throw IllegalArgumentException("Unknown view type: $viewType")
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val item = getItem(position)
        when (holder) {
            is PhotoViewHolder -> holder.bind(item)
            is VideoViewHolder -> holder.bind(item)
        }
    }

    override fun onViewRecycled(holder: RecyclerView.ViewHolder) {
        super.onViewRecycled(holder)
        if (holder is VideoViewHolder) {
            holder.releasePlayer()
        }
    }

    fun setItemLiked(contentId: String, isLiked: Boolean) {
        if (isLiked) {
            likedItems.add(contentId)
        } else {
            likedItems.remove(contentId)
        }
        // Find position and notify change
        currentList.forEachIndexed { index, feedItem ->
            if (feedItem.content_id == contentId) {
                notifyItemChanged(index)
                return
            }
        }
    }

    fun cleanup() {
        activeExoPlayers.values.forEach { it.release() }
        activeExoPlayers.clear()
    }

    inner class PhotoViewHolder(
        private val binding: ItemFeedPhotoBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(item: FeedItem) {
            // Load image
            Glide.with(binding.imageView)
                .load(item.media_url)
                .placeholder(R.drawable.placeholder_image)
                .error(R.drawable.error_image)
                .centerCrop()
                .into(binding.imageView)

            // Setup like button
            updateLikeButton(item.content_id)
            binding.likeButton.setOnClickListener {
                onLikeClick(item)
            }

            // Show category if available
            binding.categoryText.text = item.category ?: "Uncategorized"
        }

        private fun updateLikeButton(contentId: String) {
            val isLiked = likedItems.contains(contentId)
            binding.likeButton.setImageResource(
                if (isLiked) R.drawable.ic_heart_filled else R.drawable.ic_heart_outline
            )
            binding.likeButton.isEnabled = !isLiked
        }
    }

    inner class VideoViewHolder(
        private val binding: ItemFeedVideoBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        private var exoPlayer: ExoPlayer? = null
        private var currentContentId: String? = null

        fun bind(item: FeedItem) {
            currentContentId = item.content_id

            // Setup video player
            setupPlayer(item.media_url)

            // Setup like button
            updateLikeButton(item.content_id)
            binding.likeButton.setOnClickListener {
                onLikeClick(item)
            }

            // Show category if available
            binding.categoryText.text = item.category ?: "Uncategorized"

            // Load thumbnail while video loads
            Glide.with(binding.thumbnailView)
                .load(item.thumbnail_url)
                .into(binding.thumbnailView)
        }

        private fun setupPlayer(mediaUrl: String) {
            // Reuse or create player
            exoPlayer = activeExoPlayers[currentContentId] ?: ExoPlayer.Builder(itemView.context).build().also {
                activeExoPlayers[currentContentId!!] = it
            }

            binding.playerView.player = exoPlayer

            val mediaItem = MediaItem.fromUri(mediaUrl)
            exoPlayer?.apply {
                setMediaItem(mediaItem)
                prepare()
                playWhenReady = true
                volume = 0f // Mute by default
            }

            // Hide thumbnail when video starts playing
            binding.playerView.setControllerVisibilityListener { visibility ->
                binding.thumbnailView.visibility = if (visibility == View.VISIBLE) View.GONE else View.VISIBLE
            }
        }

        private fun updateLikeButton(contentId: String) {
            val isLiked = likedItems.contains(contentId)
            binding.likeButton.setImageResource(
                if (isLiked) R.drawable.ic_heart_filled else R.drawable.ic_heart_outline
            )
            binding.likeButton.isEnabled = !isLiked
        }

        fun releasePlayer() {
            exoPlayer?.release()
            currentContentId?.let { activeExoPlayers.remove(it) }
            exoPlayer = null
            currentContentId = null
        }
    }

    class FeedDiffCallback : DiffUtil.ItemCallback<FeedItem>() {
        override fun areItemsTheSame(oldItem: FeedItem, newItem: FeedItem): Boolean {
            return oldItem.content_id == newItem.content_id
        }

        override fun areContentsTheSame(oldItem: FeedItem, newItem: FeedItem): Boolean {
            return oldItem == newItem
        }
    }
}