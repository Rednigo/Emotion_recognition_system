package com.google.mediapipe.examples.facelandmarker.fragment

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.fragment.NavHostFragment
import com.google.mediapipe.examples.facelandmarker.R
import com.google.mediapipe.examples.facelandmarker.databinding.ActivityCameraBinding

class CameraActivity : AppCompatActivity() {
    private lateinit var binding: ActivityCameraBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Set up navigation to camera fragment
        val navHostFragment = supportFragmentManager
            .findFragmentById(R.id.camera_fragment_container) as NavHostFragment
        val navController = navHostFragment.navController

        // Set up back button
        binding.backButton.setOnClickListener {
            finish()
        }
    }

    override fun onBackPressed() {
        finish()
    }
}