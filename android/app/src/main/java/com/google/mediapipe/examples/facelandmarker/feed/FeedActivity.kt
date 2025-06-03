package com.google.mediapipe.examples.facelandmarker.feed

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.viewpager2.widget.ViewPager2
import com.google.mediapipe.examples.facelandmarker.databinding.ActivityFeedBinding
import com.google.mediapipe.examples.facelandmarker.network.FeedItem
import com.google.mediapipe.examples.facelandmarker.network.LikeRequest
import com.google.mediapipe.examples.facelandmarker.network.NetworkService
import com.google.mediapipe.examples.facelandmarker.utils.UserPreferencesManager
import kotlinx.coroutines.launch

class FeedActivity : AppCompatActivity() {

    private lateinit var binding: ActivityFeedBinding
    private lateinit var feedAdapter: FeedAdapter
    private lateinit var userPreferencesManager: UserPreferencesManager
    private val likedContentIds = mutableSetOf<String>()

    companion object {
        private const val TAG = "FeedActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityFeedBinding.inflate(layoutInflater)
        setContentView(binding.root)

        userPreferencesManager = UserPreferencesManager(this)

        setupUI()
        loadFeed()
    }

    private fun setupUI() {
        // Setup back button
        binding.backButton.setOnClickListener {
            finish()
        }

        // Setup ViewPager2 for swipe navigation
        feedAdapter = FeedAdapter(
            onLikeClick = { feedItem ->
                handleLikeClick(feedItem)
            }
        )

        binding.viewPager.adapter = feedAdapter
        binding.viewPager.orientation = ViewPager2.ORIENTATION_HORIZONTAL

        // Optional: Add page change callback for analytics
        binding.viewPager.registerOnPageChangeCallback(object : ViewPager2.OnPageChangeCallback() {
            override fun onPageSelected(position: Int) {
                super.onPageSelected(position)
                Log.d(TAG, "Page selected: $position")
            }
        })
    }

    private fun loadFeed() {
        binding.progressBar.visibility = View.VISIBLE
        binding.errorText.visibility = View.GONE

        lifecycleScope.launch {
            try {
                val userId = userPreferencesManager.getUserId()
                val feedItems = NetworkService.recommendationApi.getRecommendations(userId)

                if (feedItems.isEmpty()) {
                    showError("No content available")
                } else {
                    feedAdapter.submitList(feedItems)
                    binding.progressBar.visibility = View.GONE
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading feed", e)
                showError("Failed to load feed: ${e.message}")
            }
        }
    }

    private fun handleLikeClick(feedItem: FeedItem) {
        // Check if already liked
        if (likedContentIds.contains(feedItem.content_id)) {
            Toast.makeText(this, "Already liked!", Toast.LENGTH_SHORT).show()
            return
        }

        // Optimistically update UI
        likedContentIds.add(feedItem.content_id)
        feedAdapter.setItemLiked(feedItem.content_id, true)

        // Send like request
        lifecycleScope.launch {
            try {
                val request = LikeRequest(
                    user_id = userPreferencesManager.getUserId(),
                    content_id = feedItem.content_id
                )

                val response = NetworkService.recommendationApi.likeContent(
                    feedItem.content_id,
                    request
                )

                if (!response.success) {
                    // Rollback if failed
                    likedContentIds.remove(feedItem.content_id)
                    feedAdapter.setItemLiked(feedItem.content_id, false)
                    Toast.makeText(
                        this@FeedActivity,
                        "Failed to like: ${response.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            } catch (e: Exception) {
                // Rollback on error
                likedContentIds.remove(feedItem.content_id)
                feedAdapter.setItemLiked(feedItem.content_id, false)
                Toast.makeText(
                    this@FeedActivity,
                    "Error: ${e.message}",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun showError(message: String) {
        binding.progressBar.visibility = View.GONE
        binding.errorText.visibility = View.VISIBLE
        binding.errorText.text = message
    }
}